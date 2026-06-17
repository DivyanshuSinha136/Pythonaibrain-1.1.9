"""
Central configuration for the AI Memory Summarization System.
All tuneable hyperparameters and flags live here.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from ..config import get_config


@dataclass
class EmbeddingConfig:
    # TF-IDF parameters
    config = get_config(path= "./config.pbcfg")
    embedding = config.embedding
    tfidf_max_features: int = embedding.tfidf_max_features
    tfidf_ngram_range: tuple = embedding.tfidf_ngram_range
    tfidf_sublinear_tf: bool = embedding.tfidf_sublinear_tf

    # PyTorch embedding dimensions (for learned embeddings)
    embed_dim: int = embedding.embed_dim
    vocab_size: int = embedding.vocab_size


@dataclass
class ClusteringConfig:
    # KMeans fallback
    config = get_config(path= "./config.pbcfg")
    clustering = config.clustering
    n_clusters: int = clustering.n_clusters
    kmeans_max_iter: int = clustering.kmeans_max_iter
    kmeans_random_state: int = clustering.kmeans_random_state

    # DBSCAN for density-based clustering
    dbscan_eps: float = clustering.dbscan_eps
    dbscan_min_samples: int = clustering.dbscan_min_samples

    # Agglomerative
    agglo_linkage: str = clustering.agglo_linkage
    agglo_distance_threshold: Optional[float] = clustering.agglo_distance_threshold


@dataclass
class ClassifierConfig:
    # Logistic Regression intent classifier
    config = get_config(path= "./config.pbcfg")
    classifier = config.classifier
    lr_max_iter: int = classifier.lr_max_iter
    lr_C: float = classifier.lr_c
    lr_solver: str = classifier.lr_solver
    lr_multi_class: str = classifier.lr_multi_class

    # Pattern match threshold
    similarity_threshold: float = classifier.similarity_threshold


@dataclass
class SummarizerConfig:
    # PyTorch autoencoder latent dim
    config = get_config(path= "./config.pbcfg")
    summarizer = config.summarizer
    latent_dim: int = summarizer.latent_dim
    hidden_dim: int = summarizer.hidden_dim
    ae_epochs: int = summarizer.ae_epochs
    ae_lr: float = summarizer.ae_lr
    ae_batch_size: int = summarizer.ae_batch_size

    # Summarization parameters
    top_patterns_per_cluster: int = summarizer.top_patterns_per_cluster
    min_cluster_size: int = summarizer.min_cluster_size


@dataclass
class SystemConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)

    model_save_dir: str = "models/checkpoints"
    log_level: str = "INFO"
    device: str = "cpu"  # "cuda" if GPU available


# Singleton default config
DEFAULT_CONFIG = SystemConfig()
