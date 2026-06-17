"""
Clustering & Classification layer:
  - MemoryClusterer: groups patterns into semantic clusters
  - IntentClassifier: sklearn LogisticRegression intent router
  - PatternMatcher: fuzzy similarity-based lookup (fallback)
"""

import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ─── Clustering ───────────────────────────────────────────────────────────────

class MemoryClusterer:
    """
    Clusters memory embeddings using multiple strategies.
    Auto-selects best n_clusters via silhouette score.
    """

    def __init__(
        self,
        method: str = "auto",       # "kmeans" | "agglomerative" | "dbscan" | "auto"
        n_clusters: int = 8,
        random_state: int = 42,
    ):
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_: Optional[np.ndarray] = None
        self.model = None
        self._best_k: int = n_clusters

    def _auto_select_k(self, X: np.ndarray, k_range=(3, 12)) -> int:
        """Grid search best k via silhouette score."""
        n = X.shape[0]
        max_k = min(k_range[1], n - 1)
        best_k, best_score = k_range[0], -1.0
        for k in range(k_range[0], max_k + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            logger.debug(f"  k={k}  silhouette={score:.4f}")
            if score > best_score:
                best_score, best_k = score, k
        logger.info(f"Auto-selected k={best_k} (silhouette={best_score:.4f})")
        return best_k

    def fit(self, X: np.ndarray) -> "MemoryClusterer":
        method = self.method

        if method == "auto":
            self._best_k = self._auto_select_k(X)
            method = "kmeans"

        if method == "kmeans":
            self.model = KMeans(
                n_clusters=self._best_k,
                random_state=self.random_state,
                n_init=20,
            )
        elif method == "agglomerative":
            self.model = AgglomerativeClustering(
                n_clusters=self._best_k,
                linkage="ward",
            )
        elif method == "dbscan":
            self.model = DBSCAN(eps=0.5, min_samples=2)
        else:
            raise ValueError(f"Unknown clustering method: {method!r}")

        self.labels_ = self.model.fit_predict(X)
        n_unique = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        logger.info(f"Clustering → {n_unique} clusters via {method}.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new points to nearest cluster centroid."""
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        # Agglomerative has no predict — use cosine nearest centroid
        centroids = self._compute_centroids(X, self.labels_)
        sims = cosine_similarity(X, centroids)
        return np.argmax(sims, axis=1)

    @staticmethod
    def _compute_centroids(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        unique = sorted(set(labels) - {-1})
        return np.array([X[labels == l].mean(axis=0) for l in unique])

    @property
    def n_clusters_found(self) -> int:
        if self.labels_ is None:
            return 0
        return len(set(self.labels_) - {-1})


# ─── Intent Classifier ────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Logistic Regression classifier trained on TF-IDF features → intent labels.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 500):
        self.clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
        )
        self.label_encoder = LabelEncoder()
        self._fitted = False
        self.classes_: List[str] = []

    def fit(self, X: np.ndarray, labels: List[str]) -> "IntentClassifier":
        from scipy.sparse import issparse
        if issparse(X):
            X = X.toarray()
        y = self.label_encoder.fit_transform(labels)
        self.classes_ = list(self.label_encoder.classes_)
        self.clf.fit(X, y)
        self._fitted = True
        logger.info(f"IntentClassifier trained on {len(self.classes_)} intent classes.")
        return self

    def predict(self, X: np.ndarray) -> List[str]:
        from scipy.sparse import issparse
        if issparse(X):
            X = X.toarray()
        y_pred = self.clf.predict(X)
        return list(self.label_encoder.inverse_transform(y_pred))

    def predict_proba(self, X: np.ndarray) -> Dict[str, float]:
        """Returns {intent: probability} for single sample."""
        from scipy.sparse import issparse
        if issparse(X):
            X = X.toarray()
        probs = self.clf.predict_proba(X)[0]
        return dict(zip(self.classes_, probs))

    def score(self, X: np.ndarray, labels: List[str]) -> float:
        from scipy.sparse import issparse
        if issparse(X):
            X = X.toarray()
        y = self.label_encoder.transform(labels)
        return self.clf.score(X, y)


# ─── Pattern Matcher (fuzzy fallback) ─────────────────────────────────────────

class PatternMatcher:
    """
    Cosine-similarity-based fuzzy matcher.
    Used as fallback when classifier confidence is low.
    """

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self._corpus_matrix = None
        self._patterns: list = []

    def index(self, X_sparse, patterns: list) -> "PatternMatcher":
        """Index TF-IDF matrix + patterns for lookup."""
        self._corpus_matrix = X_sparse
        self._patterns = patterns
        return self

    def query(self, q_vec, top_k: int = 3) -> List[Tuple[float, object]]:
        """Return top-k (score, pattern) for a query vector."""
        sims = cosine_similarity(q_vec, self._corpus_matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_idx:
            if sims[idx] >= self.threshold:
                results.append((float(sims[idx]), self._patterns[idx]))
        return results
