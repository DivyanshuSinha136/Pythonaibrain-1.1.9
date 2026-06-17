"""
Memory Summarization Engine.

Orchestrates:
  1. Load & preprocess patterns
  2. Extract TF-IDF features
  3. Train PyTorch autoencoder → dense embeddings
  4. Cluster embeddings → semantic groups
  5. Train intent classifier
  6. Generate human-readable cluster summaries
  7. Produce structured MemorySummaryReport
"""

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .core.data_pipeline import MemoryLoader, MemoryPattern, TextFeatureExtractor
from .core.clustering import MemoryClusterer, IntentClassifier, PatternMatcher
from .models.autoencoder import AutoencoderTrainer

logger = logging.getLogger(__name__)


# ─── Report data model ────────────────────────────────────────────────────────

@dataclass
class ClusterSummary:
    cluster_id: int
    dominant_intent: str
    response_types: Dict[str, int]
    pattern_count: int
    representative_inputs: List[str]
    representative_responses: List[str]
    key_tokens: List[str]
    coherence_score: float     # intra-cluster cosine similarity mean
    description: str           # human-readable paragraph


@dataclass
class SystemStats:
    total_patterns: int
    unique_intents: List[str]
    intent_distribution: Dict[str, int]
    response_type_distribution: Dict[str, int]
    error_count: int
    command_count: int
    open_action_count: int
    avg_input_length: float
    n_clusters: int
    autoencoder_final_loss: float


@dataclass
class MemorySummaryReport:
    title: str
    cluster_summaries: List[ClusterSummary]
    system_stats: SystemStats
    recommendations: List[str]
    raw_cluster_map: Dict[int, List[str]]   # cluster_id → input keys


# ─── Summarizer ───────────────────────────────────────────────────────────────

class MemorySummarizer:
    """
    End-to-end memory summarization pipeline.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.loader = MemoryLoader(skip_empty_keys=True)
        self.extractor = TextFeatureExtractor(config=self.config.get("embedding", {}))
        self.ae_trainer = AutoencoderTrainer(
            latent_dim=self.config.get("latent_dim", 48),
            hidden_dim=self.config.get("hidden_dim", 128),
            epochs=self.config.get("ae_epochs", 40),
            lr=self.config.get("ae_lr", 1e-3),
            batch_size=self.config.get("ae_batch_size", 8),
            device=self.config.get("device", "cpu"),
        )
        self.clusterer = MemoryClusterer(method="auto")
        self.intent_clf = IntentClassifier()
        self.pattern_matcher = PatternMatcher(threshold=0.60)

        # State populated after fit()
        self.patterns_: List[MemoryPattern] = []
        self.embeddings_: Optional[np.ndarray] = None
        self.tfidf_matrix_ = None
        self.report_: Optional[MemorySummaryReport] = None

    # ── Fit pipeline ──────────────────────────────────────────────────────────

    def fit(self, raw_memory: Dict[str, str]) -> "MemorySummarizer":
        logger.info("=== MemorySummarizer: Starting fit pipeline ===")

        # 1. Load & clean
        self.patterns_ = self.loader.from_dict(raw_memory)
        n = len(self.patterns_)
        logger.info(f"Step 1: {n} patterns loaded.")

        # 2. TF-IDF features
        self.tfidf_matrix_ = self.extractor.fit_transform(self.patterns_)
        logger.info(f"Step 2: TF-IDF matrix {self.tfidf_matrix_.shape}")

        # 3. PyTorch autoencoder → dense latent embeddings
        X_dense = self.tfidf_matrix_.toarray() if hasattr(self.tfidf_matrix_, "toarray") else self.tfidf_matrix_
        self.ae_trainer.fit(X_dense)
        self.embeddings_ = self.ae_trainer.get_embeddings(X_dense)
        logger.info(f"Step 3: Embeddings shape {self.embeddings_.shape}")

        # 4. Cluster
        self.clusterer.fit(self.embeddings_)
        logger.info(f"Step 4: {self.clusterer.n_clusters_found} clusters found.")

        # 5. Train intent classifier on TF-IDF + intent labels
        intent_labels = [p.intent_tag for p in self.patterns_]
        self.intent_clf.fit(self.tfidf_matrix_, intent_labels)
        acc = self.intent_clf.score(self.tfidf_matrix_, intent_labels)
        logger.info(f"Step 5: Intent classifier train accuracy = {acc:.3f}")

        # 6. Index for fuzzy matching
        self.pattern_matcher.index(self.tfidf_matrix_, self.patterns_)

        # 7. Build report
        self.report_ = self._build_report()
        logger.info("=== MemorySummarizer: Pipeline complete ===")
        return self

    # ── Query interface ───────────────────────────────────────────────────────

    def query(self, text: str) -> Dict[str, Any]:
        """Route a new query through the pipeline and return match info."""
        from core.data_pipeline import clean_text, MemoryPattern
        p = MemoryPattern(input_text=text, response_text="", input_clean=clean_text(text))
        vec = self.extractor.transform([p])

        # Intent prediction
        intent_probs = self.intent_clf.predict_proba(vec)
        top_intent = max(intent_probs, key=intent_probs.get)

        # Cluster assignment
        emb = self.ae_trainer.get_embeddings(vec.toarray())
        cluster_id = int(self.clusterer.predict(emb)[0])

        # Fuzzy top matches
        matches = self.pattern_matcher.query(vec, top_k=3)

        return {
            "input": text,
            "predicted_intent": top_intent,
            "intent_confidence": round(intent_probs[top_intent], 3),
            "cluster_id": cluster_id,
            "top_matches": [
                {"score": round(s, 3), "input": m.input_text, "response": m.response_text}
                for s, m in matches
            ],
        }

    # ── Report builder ────────────────────────────────────────────────────────

    def _build_report(self) -> MemorySummaryReport:
        labels = self.clusterer.labels_
        n_clusters = self.clusterer.n_clusters_found

        # Group patterns by cluster
        cluster_map: Dict[int, List[MemoryPattern]] = defaultdict(list)
        for pat, label in zip(self.patterns_, labels):
            cluster_map[int(label)].append(pat)

        cluster_summaries = []
        for cid in sorted(cluster_map.keys()):
            if cid == -1:
                continue  # noise in DBSCAN
            summary = self._summarize_cluster(cid, cluster_map[cid])
            cluster_summaries.append(summary)

        stats = self._compute_stats(labels, n_clusters)
        recommendations = self._generate_recommendations(cluster_summaries, stats)

        raw_map = {
            cid: [p.input_text for p in pats]
            for cid, pats in cluster_map.items()
            if cid != -1
        }

        return MemorySummaryReport(
            title="AI Memory Pattern Summarization Report",
            cluster_summaries=cluster_summaries,
            system_stats=stats,
            recommendations=recommendations,
            raw_cluster_map=raw_map,
        )

    def _summarize_cluster(self, cid: int, patterns: List[MemoryPattern]) -> ClusterSummary:
        # Dominant intent
        intent_cnt = Counter(p.intent_tag for p in patterns)
        dominant_intent = intent_cnt.most_common(1)[0][0]

        # Response type dist
        resp_dist = dict(Counter(p.response_type for p in patterns))

        # Representative samples (shortest, most "typical")
        sorted_pats = sorted(patterns, key=lambda p: len(p.input_text))
        reps = sorted_pats[:3]

        # Key tokens from inputs
        all_tokens = " ".join(p.input_clean for p in patterns).split()
        token_freq = Counter(t for t in all_tokens if len(t) > 2)
        key_tokens = [tok for tok, _ in token_freq.most_common(8)]

        # Intra-cluster coherence
        indices = [i for i, l in enumerate(self.clusterer.labels_) if int(l) == cid]
        if len(indices) > 1:
            sub_embs = self.embeddings_[indices]
            from sklearn.metrics.pairwise import cosine_similarity
            sim_mat = cosine_similarity(sub_embs)
            np.fill_diagonal(sim_mat, 0)
            coherence = float(sim_mat.sum() / (len(indices) * (len(indices) - 1)))
        else:
            coherence = 1.0

        description = self._generate_description(
            cid, dominant_intent, resp_dist, patterns, key_tokens, coherence
        )

        return ClusterSummary(
            cluster_id=cid,
            dominant_intent=dominant_intent,
            response_types=resp_dist,
            pattern_count=len(patterns),
            representative_inputs=[p.input_text for p in reps],
            representative_responses=[p.response_text for p in reps],
            key_tokens=key_tokens,
            coherence_score=round(coherence, 4),
            description=description,
        )

    @staticmethod
    def _generate_description(
        cid, intent, resp_dist, patterns, tokens, coherence
    ) -> str:
        n = len(patterns)
        dominant_resp = max(resp_dist, key=resp_dist.get) if resp_dist else "unknown"
        token_str = ", ".join(tokens[:5]) if tokens else "various"
        coh_label = "high" if coherence > 0.5 else "moderate" if coherence > 0.2 else "low"

        desc_map = {
            "greeting": f"This cluster (#{cid}) handles greeting and opening exchanges. "
                        f"It contains {n} patterns where users initiate conversation. "
                        f"Key signals: [{token_str}]. Coherence is {coh_label}.",
            "farewell": f"Cluster #{cid} covers farewell and goodbye interactions ({n} patterns). "
                        f"Predominantly '{dominant_resp}' responses. Tokens: [{token_str}].",
            "humor": f"Cluster #{cid} groups joke and humor requests ({n} patterns). "
                     f"Users trigger these with words like [{token_str}]. {coh_label.capitalize()} semantic coherence.",
            "identity_query": f"Cluster #{cid} addresses identity and self-description queries ({n} patterns). "
                               f"Users ask who/what the AI is. Key terms: [{token_str}].",
            "command_trigger": f"Cluster #{cid} handles command invocation patterns ({n} patterns). "
                                f"These map user phrases to system actions. Tokens: [{token_str}].",
            "knowledge_query": f"Cluster #{cid} contains knowledge and information requests ({n} patterns). "
                                f"Users ask about topics via [{token_str}]. Response type: {dominant_resp}.",
            "acknowledgment": f"Cluster #{cid} groups acknowledgment and affirmation patterns ({n} patterns). "
                               f"Short confirmatory inputs like [{token_str}].",
        }
        return desc_map.get(
            intent,
            f"Cluster #{cid} is a general-purpose group ({n} patterns, intent={intent!r}). "
            f"Key terms: [{token_str}]. Coherence: {coh_label}.",
        )

    def _compute_stats(self, labels: np.ndarray, n_clusters: int) -> SystemStats:
        patterns = self.patterns_
        intent_dist = dict(Counter(p.intent_tag for p in patterns))
        resp_dist = dict(Counter(p.response_type for p in patterns))
        avg_len = float(np.mean([len(p.input_text) for p in patterns]))
        final_loss = self.ae_trainer.train_losses[-1] if self.ae_trainer.train_losses else 0.0

        return SystemStats(
            total_patterns=len(patterns),
            unique_intents=list(set(p.intent_tag for p in patterns)),
            intent_distribution=intent_dist,
            response_type_distribution=resp_dist,
            error_count=sum(1 for p in patterns if p.has_error),
            command_count=sum(1 for p in patterns if p.is_command),
            open_action_count=sum(1 for p in patterns if p.is_open_action),
            avg_input_length=round(avg_len, 2),
            n_clusters=n_clusters,
            autoencoder_final_loss=round(final_loss, 6),
        )

    @staticmethod
    def _generate_recommendations(
        summaries: List[ClusterSummary], stats: SystemStats
    ) -> List[str]:
        recs = []

        if stats.error_count > 0:
            recs.append(
                f"⚠️  {stats.error_count} error pattern(s) detected (network/API failures). "
                "Consider adding graceful fallback responses for offline scenarios."
            )

        low_coh = [s for s in summaries if s.coherence_score < 0.2]
        if low_coh:
            ids = [str(s.cluster_id) for s in low_coh]
            recs.append(
                f"🔀  Clusters {', '.join(ids)} have low semantic coherence. "
                "Consider splitting or adding more varied training examples."
            )

        small = [s for s in summaries if s.pattern_count <= 2]
        if small:
            recs.append(
                f"📉  {len(small)} cluster(s) have ≤2 patterns. "
                "Expand these clusters with additional input variations for robustness."
            )

        if stats.open_action_count > 0:
            recs.append(
                f"🔗  {stats.open_action_count} patterns trigger external 'OPEN' actions. "
                "Verify these external links/documents are still accessible."
            )

        humor = [s for s in summaries if s.dominant_intent == "humor"]
        if humor and humor[0].pattern_count > 5:
            recs.append(
                "😄  Humor cluster is well-populated. Consider adding randomization to "
                "avoid repetitive joke responses."
            )

        if stats.n_clusters > 10:
            recs.append(
                "🗂️  High cluster count detected. Consider consolidating similar intents "
                "for a leaner memory footprint."
            )

        if not recs:
            recs.append("✅  Memory patterns look well-structured and coherent!")

        return recs

    # ── Export ────────────────────────────────────────────────────────────────

    def export_report(self, path: str):
        assert self.report_ is not None, "Call fit() first."
        data = asdict(self.report_)
        Path(path).write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Report saved → {path}")

    def print_report(self):
        r = self.report_
        if r is None:
            print("No report. Run fit() first.")
            return

        print("\n" + "═" * 70)
        print(f"  {r.title}")
        print("═" * 70)

        s = r.system_stats
        print(f"\n📊  SYSTEM STATS")
        print(f"   Total patterns : {s.total_patterns}")
        print(f"   Unique intents : {', '.join(s.unique_intents)}")
        print(f"   Clusters found : {s.n_clusters}")
        print(f"   AE final loss  : {s.autoencoder_final_loss}")
        print(f"   Errors/Cmds    : {s.error_count} / {s.command_count}")
        print(f"   Intent dist    : {s.intent_distribution}")

        print(f"\n🗂️   CLUSTER SUMMARIES ({len(r.cluster_summaries)} clusters)")
        for cs in r.cluster_summaries:
            print(f"\n  ┌─ Cluster #{cs.cluster_id}  [{cs.dominant_intent}]  "
                  f"({cs.pattern_count} patterns, coherence={cs.coherence_score})")
            print(f"  │  {cs.description}")
            print(f"  │  Response types: {cs.response_types}")
            print(f"  │  Key tokens: {cs.key_tokens}")
            for inp, resp in zip(cs.representative_inputs, cs.representative_responses):
                resp_short = resp[:60] + "…" if len(resp) > 60 else resp
                print(f"  │    • {inp!r}  →  {resp_short!r}")
            print("  └" + "─" * 60)

        print(f"\n💡  RECOMMENDATIONS")
        for rec in r.recommendations:
            print(f"   {rec}")
        print("═" * 70 + "\n")
