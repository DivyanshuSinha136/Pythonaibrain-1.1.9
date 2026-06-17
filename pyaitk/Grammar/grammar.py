"""
grammar_corrector_ai.py — Production-grade Grammar Correction Pipeline
=======================================================================

Architecture (three-tier, in priority order):
  1. SpaCy + rule engine  — morphological / dependency-aware rules
  2. Scikit-learn CRF-like pipeline — statistical sequence tagger for
     common error patterns (subject-verb agreement, capitalisation, etc.)
  3. Seq2Seq (PyTorch LSTM) — neural fallback for out-of-vocabulary errors

Usage
-----
    from grammar_corrector_ai import GrammarCorrector

    # One-shot
    corrector = GrammarCorrector()
    corrector.fit(my_sentence_list)
    print(corrector.correct("i wants to here more about this"))

    # Context manager (recommended — releases GPU/CPU resources on exit)
    with GrammarCorrector() as gc:
        gc.fit(sentences)
        result = gc.correct("she go to the market")

Dependencies
------------
    pip install torch spacy scikit-learn tokenizers nltk
    python -m spacy download en_core_web_sm
    python -m nltk.downloader punkt averaged_perceptron_tagger

"""

from __future__ import annotations

import json
import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

import nltk
import spacy
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class GrammarCorrectorError(Exception):
    """Base exception for all corrector errors."""


class NotFittedError(GrammarCorrectorError):
    """Raised when correction is attempted before fitting."""


class TokenizerError(GrammarCorrectorError):
    """Raised when the BPE tokenizer is missing required special tokens."""


class IntentsValidationError(GrammarCorrectorError):
    """Raised when a JSON intents file fails schema validation."""


# ---------------------------------------------------------------------------
# JSON Intents — schema, loader, validator
# ---------------------------------------------------------------------------
"""
Expected JSON schema
--------------------
Two supported layouts — pick whichever suits your data:

Layout A — flat list of correction pairs:
{
  "intents": [
    {
      "tag": "subject_verb_agreement",
      "incorrect": "she go to the market",
      "correct":   "she goes to the market"
    },
    ...
  ]
}

Layout B — grouped (one tag, multiple examples):
{
  "intents": [
    {
      "tag": "capitalisation",
      "examples": [
        { "incorrect": "i am here",  "correct": "I am here" },
        { "incorrect": "hi my name", "correct": "Hi my name" }
      ]
    },
    ...
  ]
}

Both layouts may be mixed freely inside the same file.
The "tag" field is stored as metadata and can be used to filter training data.
"""

# Type alias for a single (incorrect, correct, tag) triple
IntentTriple = tuple[str, str, str]


@dataclass
class IntentExample:
    """One training pair extracted from the JSON file."""
    incorrect: str
    correct: str
    tag: str = "general"

    def __post_init__(self) -> None:
        if not self.incorrect.strip():
            raise IntentsValidationError("'incorrect' sentence must not be empty.")
        if not self.correct.strip():
            raise IntentsValidationError("'correct' sentence must not be empty.")
        if self.incorrect.strip() == self.correct.strip():
            logger.warning(
                "Intent tag=%r has identical incorrect/correct: %r — skipping.",
                self.tag, self.incorrect,
            )


def _parse_intent_entry(entry: dict, index: int) -> list[IntentExample]:
    """
    Parse a single entry from the ``intents`` list.
    Supports both Layout A (flat pair) and Layout B (grouped examples).
    """
    if not isinstance(entry, dict):
        raise IntentsValidationError(
            f"intents[{index}] must be a JSON object, got {type(entry).__name__}."
        )

    tag: str = str(entry.get("tag", "general")).strip() or "general"
    examples: list[IntentExample] = []

    # Layout A — single pair directly on the entry
    if "incorrect" in entry or "correct" in entry:
        incorrect = entry.get("incorrect", "")
        correct = entry.get("correct", "")
        if not isinstance(incorrect, str) or not isinstance(correct, str):
            raise IntentsValidationError(
                f"intents[{index}]: 'incorrect' and 'correct' must be strings."
            )
        ex = IntentExample(incorrect=incorrect.strip(), correct=correct.strip(), tag=tag)
        if ex.incorrect != ex.correct:
            examples.append(ex)

    # Layout B — list of pairs under "examples"
    if "examples" in entry:
        raw_examples = entry["examples"]
        if not isinstance(raw_examples, list):
            raise IntentsValidationError(
                f"intents[{index}].examples must be a JSON array."
            )
        for j, pair in enumerate(raw_examples):
            if not isinstance(pair, dict):
                raise IntentsValidationError(
                    f"intents[{index}].examples[{j}] must be a JSON object."
                )
            incorrect = pair.get("incorrect", "")
            correct = pair.get("correct", "")
            if not isinstance(incorrect, str) or not isinstance(correct, str):
                raise IntentsValidationError(
                    f"intents[{index}].examples[{j}]: fields must be strings."
                )
            ex = IntentExample(
                incorrect=incorrect.strip(),
                correct=correct.strip(),
                tag=tag,
            )
            if ex.incorrect != ex.correct:
                examples.append(ex)

    if not examples and "examples" not in entry and "incorrect" not in entry:
        raise IntentsValidationError(
            f"intents[{index}] (tag={tag!r}) has neither 'incorrect'/'correct' "
            "fields nor an 'examples' list."
        )

    return examples


def load_intents(
    source: Union[str, Path, dict],
    *,
    tags: Optional[list[str]] = None,
) -> list[IntentExample]:
    """
    Load and validate a grammar-correction intents JSON file.

    Parameters
    ----------
    source:
        File path (``str`` or ``pathlib.Path``) **or** an already-parsed
        ``dict``.  Passing a dict is handy for testing without touching disk.
    tags:
        Optional allow-list of tag strings.  When supplied only intents whose
        ``tag`` appears in this list are returned.  Pass ``None`` to load all.

    Returns
    -------
    list[IntentExample]
        Validated and deduplicated training examples.

    Raises
    ------
    FileNotFoundError
        When *source* is a path that does not exist.
    IntentsValidationError
        When the JSON structure does not match the expected schema.

    Examples
    --------
    >>> examples = load_intents("intents.json")
    >>> examples = load_intents("intents.json", tags=["capitalisation", "verb_agreement"])
    >>> examples = load_intents({"intents": [{"tag": "test",
    ...     "incorrect": "i go", "correct": "I go"}]})
    """
    # --- Load raw data ---
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Intents file not found: {path}")
        with path.open(encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise IntentsValidationError(
                    f"Invalid JSON in {path}: {exc}"
                ) from exc

    # --- Top-level validation ---
    if not isinstance(data, dict):
        raise IntentsValidationError("Root of intents JSON must be an object.")
    if "intents" not in data:
        raise IntentsValidationError("Root object must contain an 'intents' key.")
    if not isinstance(data["intents"], list):
        raise IntentsValidationError("'intents' must be a JSON array.")

    # --- Parse entries ---
    all_examples: list[IntentExample] = []
    for i, entry in enumerate(data["intents"]):
        parsed = _parse_intent_entry(entry, i)
        all_examples.extend(parsed)

    # --- Tag filter ---
    if tags is not None:
        tag_set = set(tags)
        all_examples = [ex for ex in all_examples if ex.tag in tag_set]
        logger.info("Tag filter %s → %d examples retained.", tag_set, len(all_examples))

    # --- Deduplication ---
    seen: set[tuple[str, str]] = set()
    unique: list[IntentExample] = []
    for ex in all_examples:
        key = (ex.incorrect.lower(), ex.correct.lower())
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    duplicates = len(all_examples) - len(unique)
    if duplicates:
        logger.warning("Removed %d duplicate intent examples.", duplicates)

    logger.info(
        "Loaded %d intent examples across %d unique tags.",
        len(unique),
        len({ex.tag for ex in unique}),
    )
    return unique


def intents_to_pairs(examples: list[IntentExample]) -> list[tuple[str, str]]:
    """Convert ``[IntentExample, …]`` → ``[(incorrect, correct), …]`` pairs."""
    return [(ex.incorrect, ex.correct) for ex in examples]


def intents_summary(examples: list[IntentExample]) -> dict[str, int]:
    """Return a ``{tag: count}`` dict for quick inspection."""
    summary: dict[str, int] = {}
    for ex in examples:
        summary[ex.tag] = summary.get(ex.tag, 0) + 1
    return dict(sorted(summary.items()))


# ---------------------------------------------------------------------------
# NLTK / SpaCy bootstrap
# ---------------------------------------------------------------------------
def _ensure_nltk() -> None:
    for resource in ("punkt", "averaged_perceptron_tagger", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)


def _load_spacy(model: str = "en_core_web_sm") -> spacy.language.Language:
    try:
        return spacy.load(model)
    except OSError:
        logger.warning("SpaCy model '%s' not found — downloading…", model)
        from spacy.cli import download as spacy_download  # type: ignore
        spacy_download(model)
        return spacy.load(model)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CorrectorConfig:
    """All tunable knobs in one place."""

    # BPE tokenizer
    vocab_size: int = 2000
    special_tokens: list[str] = field(
        default_factory=lambda: ["<s>", "</s>", "<unk>", "<pad>"]
    )

    # Seq2Seq architecture
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3

    # Training
    epochs: int = 10
    learning_rate: float = 1e-3
    teacher_forcing_ratio: float = 0.5   # probability of using teacher forcing
    max_decode_len: int = 120

    # SpaCy model
    spacy_model: str = "en_core_web_sm"

    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Tier 1 — Rule-based corrector (SpaCy + regex + morphology)
# ---------------------------------------------------------------------------

# Irregular verb mapping: base → third-person singular
_IRREGULAR_VERBS: dict[str, str] = {
    "be": "is", "have": "has", "do": "does",
    "go": "goes", "say": "says", "make": "makes",
}

# Simple confused-word pairs (lowercase)
_CONFUSED_WORDS: dict[str, str] = {
    "to here": "to hear",
    "your welcome": "you're welcome",
    "their going": "they're going",
    "its a": "it's a",
}


def _apply_confused_words(text: str) -> str:
    for wrong, right in _CONFUSED_WORDS.items():
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
    return text


def rule_based_corrector(text: str, nlp: spacy.language.Language) -> str:
    """
    Tier-1 correction: SpaCy dependency parse + morphological rules + regex.

    Corrections applied
    -------------------
    * Lowercase ``i`` → ``I`` (pronoun capitalisation)
    * Sentence-initial capitalisation
    * Subject–verb agreement using SpaCy dependency labels and POS tags
    * Confused-word pairs (to here → to hear, etc.)
    """
    # --- regex pre-pass ---
    text = re.sub(r"\bi\b", "I", text)
    text = _apply_confused_words(text)

    # --- SpaCy morphological pass ---
    doc = nlp(text)
    tokens: list[str] = []

    for token in doc:
        word = token.text

        # Subject–verb agreement: 3rd-person singular subjects
        if (
            token.dep_ == "ROOT"
            and token.pos_ == "VERB"
            and token.lemma_ in _IRREGULAR_VERBS
        ):
            # Find the nominal subject
            subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
            if subjects:
                subj = subjects[0]
                # He/she/it → 3sg
                if subj.text.lower() in ("he", "she", "it") or (
                    subj.tag_ in ("NN", "NNP") and not any(
                        c.dep_ == "conj" for c in subj.children
                    )
                ):
                    word = _IRREGULAR_VERBS.get(token.lemma_, word)

        tokens.append(word)

    result = " ".join(tokens)

    # --- Sentence capitalisation ---
    sentences = re.split(r"(?<=[.!?])\s+", result)
    result = " ".join(s[0].upper() + s[1:] if s else s for s in sentences)
    if result:
        result = result[0].upper() + result[1:]

    return result


# ---------------------------------------------------------------------------
# Tier 2 — Scikit-learn statistical corrector
# ---------------------------------------------------------------------------

def _extract_token_features(tokens: list[str], idx: int) -> dict:
    """Feature dict for the token at position *idx*."""
    tok = tokens[idx]
    features: dict = {
        "tok.lower": tok.lower(),
        "tok.isupper": tok.isupper(),
        "tok.istitle": tok.istitle(),
        "tok.isdigit": tok.isdigit(),
        "tok.suffix2": tok.lower()[-2:],
        "tok.suffix3": tok.lower()[-3:],
        "tok.prefix2": tok.lower()[:2],
        "BOS": idx == 0,
        "EOS": idx == len(tokens) - 1,
    }
    if idx > 0:
        prev = tokens[idx - 1]
        features.update({
            "prev.lower": prev.lower(),
            "prev.istitle": prev.istitle(),
        })
    if idx < len(tokens) - 1:
        nxt = tokens[idx + 1]
        features.update({
            "next.lower": nxt.lower(),
            "next.istitle": nxt.istitle(),
        })
    return features


class TokenFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that converts a list of token sequences
    into per-token feature dicts.
    """

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        return [
            [_extract_token_features(seq, i) for i in range(len(seq))]
            for seq in X
        ]


def build_sklearn_pipeline() -> Pipeline:
    """
    Return a sklearn Pipeline ready to be fitted on (corrupted, correct) pairs.

    Currently used to detect capitalisation errors and simple token-swap
    patterns via a DictVectorizer + LogisticRegression sequence model.
    The pipeline operates token-by-token and outputs a corrected/unchanged
    label per token.
    """
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    # We flatten sequences for fitting, then reshape at inference.
    return Pipeline([
        ("features", DictVectorizer(sparse=True)),
        ("clf", OneVsRestClassifier(
            LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        )),
    ])


def _flatten_for_sklearn(
    pairs: list[tuple[list[str], list[str]]]
) -> tuple[list[dict], list[str]]:
    """Flatten (corrupted_tokens, correct_tokens) pairs into (X, y)."""
    X, y = [], []
    for corrupt_toks, correct_toks in pairs:
        length = min(len(corrupt_toks), len(correct_toks))
        for i in range(length):
            X.append(_extract_token_features(corrupt_toks, i))
            y.append(correct_toks[i])
    return X, y


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

_CORRUPTION_RULES: list[tuple[str, str]] = [
    # --- Original & Basic Case ---
    ("I", "i"),
    ("hear", "here"),
    (r"\bgoes\b", "go"),
    (r"\bHi\b", "hi"),
    (r"Thanks, I", "thanks i"),
    
    # --- Homophones & Common Mixups (30) ---
    (r"\btheir\b", "there"), (r"\bthere\b", "they're"), (r"\bthey're\b", "their"),
    (r"\byour\b", "you're"), (r"\byou're\b", "your"),
    (r"\bits\b", "it's"), (r"\bit's\b", "its"),
    (r"\btoo\b", "to"), (r"\bto\b", "2"),
    (r"\blose\b", "loose"), (r"\bloose\b", "lose"),
    (r"\baffect\b", "effect"), (r"\beffect\b", "affect"),
    (r"\bweather\b", "whether"), (r"\bwhether\b", "weather"),
    (r"\bwhose\b", "who's"), (r"\bwho's\b", "whose"),
    (r"\ballowed\b", "aloud"), (r"\baloud\b", "allowed"),
    (r"\bbuy\b", "by"), (r"\bby\b", "buy"),
    (r"\bknow\b", "no"), (r"\bknew\b", "new"),
    (r"\bpassed\b", "past"), (r"\bpast\b", "passed"),
    (r"\bthrough\b", "thru"), (r"\bright\b", "rite"),
    (r"\bwould have\b", "would of"), (r"\bcould have\b", "could of"), (r"\bshould have\b", "should of"),

    # --- Internet Slang & Shorthand (30) ---
    (r"\bbecause\b", "bc"), (r"\bpeople\b", "ppl"),
    (r"\bwith\b", "w/"), (r"\bwithout\b", "w/o"),
    (r"\bbefore\b", "b4"), (r"\bgreat\b", "gr8"),
    (r"\bfor\b", "4"), (r"\btonight\b", "2nite"),
    (r"\bplease\b", "pls"), (r"\bokay\b", "kk"),
    (r"\bthanks\b", "thx"), (r"\bmessage\b", "msg"),
    (r"\breally\b", "rly"), (r"\bare\b", "r"),
    (r"\byou\b", "u"), (r"\bwhy\b", "y"),
    (r"\bsee\b", "c"), (r"\beveryone\b", "evry1"),
    (r"\bsomeone\b", "some1"), (r"\banyone\b", "any1"),
    (r"\boh my god\b", "omg"), (r"\bby the way\b", "btw"),
    (r"\bin my opinion\b", "imo"), (r"\bfor your information\b", "fyi"),
    (r"\blaughing out loud\b", "lol"), (r"\bbe right back\b", "brb"),
    (r"\bi don't know\b", "idk"), (r"\bnever mind\b", "nvm"),
    (r"\bsee you\b", "cya"), (r"\blater\b", "l8r"),

    # --- Informal Contractions & "Lazy" Speech (20) ---
    (r"\bwant to\b", "wanna"), (r"\bgoing to\b", "gonna"),
    (r"\bgot to\b", "gotta"), (r"\bkind of\b", "kinda"),
    (r"\bsort of\b", "sorta"), (r"\blet me\b", "lemme"),
    (r"\bgive me\b", "gimme"), (r"\bdon't know\b", "dunno"),
    (r"\bout of\b", "outta"), (r"\bhave to\b", "hafta"),
    (r"\bprobably\b", "prolly"), (r"\bam not\b", "aint"),
    (r"\bis not\b", "aint"), (r"\bare not\b", "aint"),
    (r"\bthem\b", "'em"), (r"\babout\b", "'bout"),
    (r"\band\b", "n"), (r"\bcome on\b", "c'mon"),
    (r"\bevery\b", "evry"), (r"\bworking\b", "workin"),

    # --- Common Typos & Keyboard Slips (20) ---
    (r"\bthe\b", "teh"), (r"\band\b", "adn"),
    (r"\bjust\b", "jsut"), (r"\bfrom\b", "fom"),
    (r"\bhelp\b", "hlp"), (r"\bfriend\b", "freind"),
    (r"\breceive\b", "recieve"), (r"\bbelieve\b", "beleive"),
    (r"\btomorrow\b", "tommorow"), (r"\bgovernment\b", "govment"),
    (r"\bdefinitely\b", "defiantly"), (r"\bseparate\b", "seperate"),
    (r"\buntil\b", "util"), (r"\bbeautiful\b", "beatiful"),
    (r"\bawesome\b", "awsome"), (r"\benough\b", "enuf"),
    (r"\bschool\b", "skool"), (r"\bphone\b", "fone"),
    (r"\bnight\b", "nite"), (r"\bthought\b", "thot"),

    # --- Punctuation & Style Corruption (5) ---
    (r"\!", "!!1!"),
    (r"\?", "??"),
    (r"\bI am\b", "im"),
    (r"\s+", "  "), # Accidental double spacing
    (r"\.\.\.", ".."), # Improper ellipses
]

def corrupt_sentence(sentence: str) -> str:
    """Apply deterministic corruption rules to produce a noisy sentence."""
    for pattern, replacement in _CORRUPTION_RULES:
        sentence = re.sub(pattern, replacement, sentence)
    return sentence


def build_dataset(
    correct_sentences: Iterable[str],
) -> list[tuple[str, str]]:
    """Return ``[(corrupted, correct), …]`` pairs."""
    return [(corrupt_sentence(s), s) for s in correct_sentences]


# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------

def train_tokenizer(sentences: Iterable[str]) -> Tokenizer:
    """Train and return a BPE tokenizer on *sentences*."""
    cfg = CorrectorConfig()
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        special_tokens=cfg.special_tokens,
        show_progress=False,
    )
    tokenizer.train_from_iterator(list(sentences), trainer)
    logger.info("BPE tokenizer trained — vocab size: %d", tokenizer.get_vocab_size())
    return tokenizer


def _token_id(tokenizer: Tokenizer, tok: str) -> int:
    tid = tokenizer.token_to_id(tok)
    if tid is None:
        raise TokenizerError(f"Tokenizer is missing special token: {tok!r}")
    return tid


# ---------------------------------------------------------------------------
# Seq2Seq model (Tier 3)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            emb_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor):
        embedded = self.dropout(self.embedding(src))
        _, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            emb_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        embedded = self.dropout(self.embedding(tgt))          # (B, 1, E)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        logits = self.fc(output[:, -1])                        # (B, V)
        return logits, hidden, cell


class Seq2Seq(nn.Module):
    """LSTM Encoder-Decoder with teacher forcing and configurable depth."""

    def __init__(self, config: CorrectorConfig, vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            vocab_size, config.emb_dim, config.hidden_dim,
            config.num_layers, config.dropout,
        )
        self.decoder = Decoder(
            vocab_size, config.emb_dim, config.hidden_dim,
            config.num_layers, config.dropout,
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)

        hidden, cell = self.encoder(src)
        dec_input = tgt[:, 0:1]          # <s>

        for t in range(1, tgt_len):
            logits, hidden, cell = self.decoder(dec_input, hidden, cell)
            outputs[:, t] = logits
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = tgt[:, t:t+1] if use_teacher else logits.argmax(-1, keepdim=True)

        return outputs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: Seq2Seq,
    data: list[tuple[str, str]],
    tokenizer: Tokenizer,
    config: Optional[CorrectorConfig] = None,
) -> list[float]:
    """
    Train *model* on *(corrupted, correct)* pairs.

    Returns
    -------
    list[float]
        Per-epoch average loss values.
    """
    config = config or model.config
    device = torch.device(config.device)
    model.to(device).train()

    pad_id = _token_id(tokenizer, "<pad>")
    sos_id = _token_id(tokenizer, "<s>")
    eos_id = _token_id(tokenizer, "</s>")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    epoch_losses: list[float] = []

    for epoch in range(config.epochs):
        total_loss = 0.0

        for corrupted, correct in data:
            src_ids = [sos_id] + tokenizer.encode(corrupted).ids + [eos_id]
            tgt_ids = [sos_id] + tokenizer.encode(correct).ids + [eos_id]

            src = torch.tensor([src_ids], device=device)
            tgt = torch.tensor([tgt_ids], device=device)

            optimizer.zero_grad()
            output = model(src, tgt, config.teacher_forcing_ratio)

            # output: (1, tgt_len, vocab) — ignore the <s> step
            out_flat = output[:, 1:].reshape(-1, output.size(-1))
            tgt_flat = tgt[:, 1:].reshape(-1)
            loss = criterion(out_flat, tgt_flat)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(data), 1)
        scheduler.step(avg_loss)
        epoch_losses.append(avg_loss)
        logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, config.epochs, avg_loss)

    return epoch_losses


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def correct_sentence_neural(
    model: Seq2Seq,
    input_text: str,
    tokenizer: Tokenizer,
    config: Optional[CorrectorConfig] = None,
) -> str:
    """Run greedy decoding on *input_text* using the trained Seq2Seq model."""
    config = config or model.config
    device = torch.device(config.device)
    model.to(device).eval()

    sos_id = _token_id(tokenizer, "<s>")
    eos_id = _token_id(tokenizer, "</s>")

    src_ids = tokenizer.encode(str(input_text)).ids
    src = torch.tensor([src_ids], device=device)

    hidden, cell = model.encoder(src)
    dec_input = torch.tensor([[sos_id]], device=device)
    output_ids: list[int] = []

    for _ in range(config.max_decode_len):
        logits, hidden, cell = model.decoder(dec_input, hidden, cell)
        predicted_id: int = logits.argmax(dim=-1).item()
        if predicted_id == eos_id:
            break
        output_ids.append(predicted_id)
        dec_input = torch.tensor([[predicted_id]], device=device)

    return tokenizer.decode(output_ids)


# ---------------------------------------------------------------------------
# Unified GrammarCorrector facade
# ---------------------------------------------------------------------------

class GrammarCorrector:
    """
    Three-tier grammar correction pipeline.

    Tier 1 (SpaCy + rules) always runs.
    Tier 2 (sklearn) runs if fitted; improves token-level statistical fixes.
    Tier 3 (Seq2Seq) runs if fitted; handles complex / unseen patterns.

    Context Manager
    ---------------
    >>> with GrammarCorrector() as gc:
    ...     gc.fit(sentences)
    ...     print(gc.correct("she go to market"))
    """

    def __init__(self, config: Optional[CorrectorConfig] = None) -> None:
        _ensure_nltk()
        self.config = config or CorrectorConfig()
        self._nlp: Optional[spacy.language.Language] = None
        self._tokenizer: Optional[Tokenizer] = None
        self._seq2seq: Optional[Seq2Seq] = None
        self._sklearn_pipe: Optional[Pipeline] = None
        self._fitted: bool = False
        logger.debug("GrammarCorrector created — device: %s", self.config.device)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "GrammarCorrector":
        self._nlp = _load_spacy(self.config.spacy_model)
        logger.debug("SpaCy model loaded.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Release heavy resources
        self._nlp = None
        self._seq2seq = None
        self._sklearn_pipe = None
        self._tokenizer = None
        logger.debug("GrammarCorrector resources released.")
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        sentences: list[str],
        *,
        intents: Optional[list[IntentExample]] = None,
    ) -> "GrammarCorrector":
        """
        Train all three tiers.

        Parameters
        ----------
        sentences:
            A list of *correct* sentences. Corrupted versions are generated
            automatically via ``build_dataset()``.
        intents:
            Optional list of ``IntentExample`` objects (loaded via
            ``load_intents()``).  When supplied their (incorrect, correct)
            pairs are merged with the auto-corrupted sentence pairs before
            training, giving the model explicit examples of real-world errors.

        Returns
        -------
        self (for chaining)

        Examples
        --------
        >>> examples = load_intents("intents.json")
        >>> gc.fit([], intents=examples)           # intents-only training
        >>> gc.fit(sentences, intents=examples)    # merge both sources
        """
        if self._nlp is None:
            self._nlp = _load_spacy(self.config.spacy_model)

        # --- Build base dataset from auto-corrupted correct sentences ---
        dataset: list[tuple[str, str]] = build_dataset(sentences)

        # --- Merge explicit intent pairs ---
        if intents:
            intent_pairs = intents_to_pairs(intents)
            dataset = dataset + intent_pairs
            summary = intents_summary(intents)
            logger.info(
                "Merged %d intent examples into training set. Tag breakdown: %s",
                len(intent_pairs), summary,
            )

        if not dataset:
            raise GrammarCorrectorError(
                "No training data: provide at least one sentence or one intent example."
            )

        all_texts = [t for pair in dataset for t in pair]

        # --- BPE tokenizer ---
        logger.info("Training BPE tokenizer…")
        self._tokenizer = train_tokenizer(all_texts)

        # --- Sklearn tier ---
        logger.info("Fitting sklearn pipeline…")
        token_pairs = [
            (word_tokenize(corrupt), word_tokenize(correct))
            for corrupt, correct in dataset
        ]
        X_flat, y_flat = _flatten_for_sklearn(token_pairs)
        if X_flat:
            self._sklearn_pipe = build_sklearn_pipeline()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._sklearn_pipe.fit(X_flat, y_flat)
            logger.info("Sklearn pipeline fitted on %d token samples.", len(X_flat))

        # --- Seq2Seq tier ---
        logger.info("Training Seq2Seq model (%d epochs)…", self.config.epochs)
        vocab_size = self._tokenizer.get_vocab_size()
        self._seq2seq = Seq2Seq(self.config, vocab_size)
        train_model(self._seq2seq, dataset, self._tokenizer, self.config)

        self._fitted = True
        logger.info("GrammarCorrector fit complete.")
        return self

    def correct(self, text: str) -> str:
        """
        Apply all available correction tiers to *text* and return the result.

        Tier 1 always runs (no fitting required).
        Tiers 2 and 3 run only after ``fit()`` has been called.

        Raises
        ------
        GrammarCorrectorError — if SpaCy model failed to load.
        """
        if self._nlp is None:
            self._nlp = _load_spacy(self.config.spacy_model)

        # Tier 1 — rule-based
        result = rule_based_corrector(text, self._nlp)
        logger.debug("Tier-1 output: %r", result)

        if not self._fitted:
            return result

        # Tier 2 — sklearn token-level fix
        result = self._sklearn_correct(result)
        logger.debug("Tier-2 output: %r", result)

        # Tier 3 — neural seq2seq
        if self._seq2seq is not None and self._tokenizer is not None:
            try:
                neural_out = correct_sentence_neural(
                    self._seq2seq, result, self._tokenizer, self.config
                )
                if neural_out.strip():
                    result = neural_out
                    logger.debug("Tier-3 output: %r", result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Seq2Seq inference failed (using Tier-2 output): %s", exc)

        return result

    def fit_from_intents(
        self,
        intents: list[IntentExample],
        *,
        sentences: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
    ) -> "GrammarCorrector":
        """
        Train exclusively (or primarily) from ``IntentExample`` objects.

        This is a convenience wrapper around ``fit()`` for when your primary
        training source is an intents file rather than a plain sentence list.

        Parameters
        ----------
        intents:
            List of ``IntentExample`` objects — typically the return value of
            ``load_intents()``.
        sentences:
            Optional extra *correct* sentences.  Auto-corrupted and merged.
        tags:
            Optional tag filter applied *before* training.  Only examples
            whose ``tag`` is in this list will be used.

        Returns
        -------
        self (for chaining)

        Examples
        --------
        >>> with GrammarCorrector() as gc:
        ...     examples = load_intents("intents.json")
        ...     gc.fit_from_intents(examples)
        ...     print(gc.correct("she go to market"))

        >>> # Filter to specific error categories
        ...     gc.fit_from_intents(examples, tags=["verb_agreement"])
        """
        if tags is not None:
            tag_set = set(tags)
            intents = [ex for ex in intents if ex.tag in tag_set]
            logger.info("Tag filter %s → %d examples for training.", tag_set, len(intents))

        return self.fit(sentences or [], intents=intents)

    def fit_from_intents_file(
        self,
        path: Union[str, Path],
        *,
        sentences: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
    ) -> "GrammarCorrector":
        """
        Load a JSON intents file and train in one call.

        Parameters
        ----------
        path:
            Path to the ``.json`` intents file.
        sentences:
            Optional extra *correct* sentences merged into training.
        tags:
            Optional tag allow-list (see ``load_intents()``).

        Returns
        -------
        self (for chaining)

        Examples
        --------
        >>> with GrammarCorrector() as gc:
        ...     gc.fit_from_intents_file("intents.json")
        ...     print(gc.correct("i wants to here you"))

        >>> # One-liner with tag filter
        ...     gc.fit_from_intents_file("intents.json", tags=["capitalisation"])
        """
        examples = load_intents(path, tags=tags)
        return self.fit_from_intents(examples, sentences=sentences)

    def save(self, path: str) -> None:
        """Persist model weights and tokenizer to *path* (directory)."""
        import os, json
        os.makedirs(path, exist_ok=True)
        if self._seq2seq:
            torch.save(self._seq2seq.state_dict(), f"{path}/seq2seq.pt")
        if self._tokenizer:
            self._tokenizer.save(f"{path}/tokenizer.json")
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> "GrammarCorrector":
        """Restore weights and tokenizer from *path* (directory)."""
        self._tokenizer = Tokenizer.from_file(f"{path}/tokenizer.json")
        vocab_size = self._tokenizer.get_vocab_size()
        self._seq2seq = Seq2Seq(self.config, vocab_size)
        self._seq2seq.load_state_dict(
            torch.load(f"{path}/seq2seq.pt", map_location=self.config.device)
        )
        self._fitted = True
        logger.info("Model loaded from %s", path)
        return self

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    def _sklearn_correct(self, text: str) -> str:
        """Apply sklearn token-level corrections."""
        if self._sklearn_pipe is None:
            return text
        tokens = word_tokenize(text)
        if not tokens:
            return text
        features = [_extract_token_features(tokens, i) for i in range(len(tokens))]
        try:
            corrected = self._sklearn_pipe.predict(features)
            return " ".join(corrected)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sklearn correction failed (using input): %s", exc)
            return text


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def quick_correct(text: str, sentences: Optional[list[str]] = None) -> str:
    """
    One-liner: correct *text* using a freshly fitted (or rule-only) corrector.

    If *sentences* are provided the full three-tier pipeline is trained first.
    Otherwise only Tier-1 (SpaCy + rules) runs.
    """
    with GrammarCorrector() as gc:
        if sentences:
            gc.fit(sentences)
        return gc.correct(text)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
__all__ = [
    # Exceptions
    "GrammarCorrectorError",
    "NotFittedError",
    "TokenizerError",
    "IntentsValidationError",
    # Config
    "CorrectorConfig",
    # Intents
    "IntentExample",
    "load_intents",
    "intents_to_pairs",
    "intents_summary",
    # Main facade
    "GrammarCorrector",
    # Data helpers
    "corrupt_sentence",
    "build_dataset",
    # Tokenizer
    "train_tokenizer",
    # Tier-1
    "rule_based_corrector",
    # Tier-3 model + helpers
    "Seq2Seq",
    "Encoder",
    "Decoder",
    "train_model",
    "correct_sentence_neural",
    # Convenience
    "quick_correct",
]


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse, sys

    parser = argparse.ArgumentParser(description="Grammar Corrector CLI")
    parser.add_argument("text", nargs="?", help="Text to correct (reads stdin if omitted)")
    parser.add_argument("--train-file", metavar="FILE",
                        help="Plain-text file with one correct sentence per line")
    parser.add_argument("--intents-file", metavar="JSON",
                        help="JSON intents file with incorrect/correct pairs")
    parser.add_argument("--tags", metavar="TAG", nargs="+",
                        help="Only use intents whose tag matches (space-separated)")
    parser.add_argument("--show-intents-summary", action="store_true",
                        help="Print tag/count breakdown of the intents file and exit")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-model", metavar="DIR", help="Save model after training")
    parser.add_argument("--load-model", metavar="DIR", help="Load pre-trained model")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Intents summary (no correction needed) ---
    if args.show_intents_summary:
        if not args.intents_file:
            parser.error("--show-intents-summary requires --intents-file.")
        examples = load_intents(args.intents_file, tags=args.tags)
        summary = intents_summary(examples)
        print(f"Intents summary ({len(examples)} examples):")
        for tag, count in summary.items():
            print(f"  {tag}: {count}")
        return

    text = args.text or sys.stdin.read().strip()
    if not text:
        parser.error("Provide text to correct as argument or via stdin.")

    cfg = CorrectorConfig(epochs=args.epochs, device=args.device)

    with GrammarCorrector(cfg) as gc:
        if args.load_model:
            gc.load(args.load_model)

        elif args.intents_file or args.train_file:
            sentences: list[str] = []
            if args.train_file:
                with open(args.train_file, encoding="utf-8") as fh:
                    sentences = [ln.strip() for ln in fh if ln.strip()]

            if args.intents_file:
                gc.fit_from_intents_file(
                    args.intents_file,
                    sentences=sentences,
                    tags=args.tags,
                )
            else:
                gc.fit(sentences)

        else:
            logger.info("No training data — running Tier-1 (rule-based) only.")

        result = gc.correct(text)
        print(result)

        if args.save_model:
            gc.save(args.save_model)


if __name__ == "__main__":
    _main()
