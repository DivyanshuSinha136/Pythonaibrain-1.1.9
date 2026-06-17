"""
core.py  –  Production-grade PythonAIBrain core module.

Changelog
---------
Memory.py integration (this revision)
  • Brain.__init__ now accepts memory_path, smart_memory, and
    memory_fit_interval kwargs and wires them through build_memory().
  • SmartMemory is used by default (smart_memory=True); falls back to
    plain Memory transparently when summarizer.py is absent.
  • Redundant load_memory() calls before remember() removed from
    process_messages, memorize_user_name, and recall_user_name — the
    SmartMemory/Memory contract guarantees the in-memory store is always
    current after __init__; explicit load() is only needed on cold start.
  • New Brain methods: search_memory(), memory_intent(),
    memory_report(), export_memory_report(), fit_memory().
  • SmartMemory and build_memory added to __all__.

JAX removed (previous revision)
  JAX and XLA were the source of repeated instability on Windows:
    • TPU backend errors on every startup ("Unable to initialize backend 'tpu'")
    • static_argnums tracing failures when vocab_size was passed as a runtime
      value to jnp.zeros() — XLA requires shapes to be compile-time constants
    • Windows support gaps across JAX versions

  The BoW matrix is now a straightforward NumPy operation.  NumPy's
  vectorised indexing has the same O(n·v) complexity as the JAX vmap path
  for the vocabulary sizes used here (typically < 10 000 tokens), and it
  introduces zero extra dependencies, zero JIT warm-up latency, and zero
  risk of silent trace-time failures hiding the real error.

Previous fixes (retained)
  • @dataclass removed from Brain / AdvanceBrain / ChatbotAssistant — the
    decorator's generated __init__ was silently overwriting the hand-written
    one and nullifying __enter__ / __exit__.
  • pythonaibrain_llm replaces the removed .LLMs.TIGER import.
  • Weather URL bug fixed (urlencode used for all query params).
  • All bare except: replaced with typed handlers + structured logging.
  • translate_to_en / predict_frame / predictFrameAdvance singletons so
    heavy models train only once per process.
  • _write_message word-index guard against IndexError on the last word.
  • process_message guards model-is-None before running inference.
  • torch._dynamo / torch._inductor access wrapped in try/except for older
    PyTorch builds that don't expose those attributes.

Usage:
    from pyaitk.core import Brain, VectorizerMode

    # TF-IDF (sklearn)
    brain = Brain(vectorizer_mode=VectorizerMode.TFIDF)

    # Gensim TF-IDF (pip install gensim)
    brain = Brain(vectorizer_mode=VectorizerMode.GENSIM)

    # Original binary BoW (default — zero behaviour change)
    brain = Brain()  # or VectorizerMode.BOW

"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import json
import logging
import os
import random
import sys
import webbrowser
from collections import Counter
from enum import Enum
from importlib import resources
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as urllib_request
from urllib.error import URLError
from urllib.parse import urlencode

# ── third-party ───────────────────────────────────────────────────────────────
import nltk
import numpy as np
import psutil
import pyjokes
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ── Gensim (optional) ─────────────────────────────────────────────────────────
try:
    from gensim import corpora, models as gensim_models
    from gensim.models import TfidfModel as GensimTfidfModel
    _GENSIM_AVAILABLE: bool = True
except ImportError:
    _GENSIM_AVAILABLE = False

# ── local ─────────────────────────────────────────────────────────────────────
from .config import AppConfig, get_config
from .Grammar import GrammarCorrector
from .Memory import Memory, SmartMemory, build_memory
from .Search import Search
from .TTS import speak
from .NER import NERPipeline, NERTrainer
from .NER.trainer import TrainerConfig

torch._dynamo.config.suppress_errors = True

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── pythonaibrain-llm (optional) ─────────────────────────────────────────────
try:
    from pythonaibrain_llm import load_llm_model, validate_system
    _PYAI_LLM_AVAILABLE: bool = True
except ImportError:
    _PYAI_LLM_AVAILABLE = False
    logger.warning(
        "pythonaibrain-llm not installed – AdvanceBrain LLM features disabled. "
        "Install with: pip install pythonaibrain-llm"
    )

# ── torch compiler flags (optional, silently ignored on older builds) ─────────
try:
    torch._dynamo.config.suppress_errors = True   # type: ignore[attr-defined]
    torch._inductor.config.max_autotune = True    # type: ignore[attr-defined]
except AttributeError:
    pass

load_dotenv()
_WEATHER_API_KEY: str = os.getenv("weather_api_key", "")

# ── config layer ──────────────────────────────────────────────────────────────
# Auto-discover a config.pbcfg file (cwd → home). All subsystems read their
# settings from this singleton; users can override by calling configure().
_cfg: "AppConfig" = get_config()

_WEATHER_BASE_URL: str = _cfg.weather.base_url


def configure(path: str) -> "AppConfig":
    """
    Load a specific .pbcfg file and make it the active configuration.

    Call this *before* constructing Brain / AdvanceBrain to have all
    settings applied from your custom config file.

    Parameters
    ----------
    path : str
        Path to a .pbcfg file.

    Returns
    -------
    AppConfig
        The newly loaded config (also accessible via core._cfg).

    Example
    -------
        import pythonaibrain.core as core
        core.configure("project.pbcfg")
        brain = core.Brain()
    """
    global _cfg, _WEATHER_BASE_URL
    _cfg = get_config(path)
    _WEATHER_BASE_URL = _cfg.weather.base_url
    _apply_logging_config(_cfg)
    logger.info("PythonAIBrain reconfigured from %s", path)
    return _cfg


def _apply_logging_config(cfg: "AppConfig") -> None:
    """Apply [logging] section from config to the root logger."""
    level_str = cfg.logging.level.upper()
    level = getattr(logging, level_str, logging.INFO)
    logging.basicConfig(level=level, format=cfg.logging.format, force=True)
    logger.setLevel(level)


# Apply logging settings from config on import
_apply_logging_config(_cfg)


__all__ = ["Brain", "AdvanceBrain", "IntentsManager", "get_weather",
           "Memory", "SmartMemory", "build_memory",
           "AppConfig", "configure", "get_config",
           "VectorizerMode"]


# ─────────────────────────────────────────────────────────────────────────────
# Vectorizer mode enum
# ─────────────────────────────────────────────────────────────────────────────

class VectorizerMode(str, Enum):
    """
    Feature-extraction strategy for intent classification.

    BOW    – Binary Bag-of-Words (original behaviour, pure NumPy).
    TFIDF  – TF-IDF weighting via scikit-learn's TfidfVectorizer.
    GENSIM – TF-IDF via Gensim's Dictionary + TfidfModel (requires gensim).

    Usage::

        from pythonaibrain.core import Brain, VectorizerMode
        brain = Brain(vectorizer_mode=VectorizerMode.TFIDF)
    """
    BOW    = "bow"
    TFIDF  = "tfidf"
    GENSIM = "gensim"


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix builders — BoW (NumPy), TF-IDF (sklearn), Gensim TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

def _build_bow_matrix(
    documents: List[Tuple[List[str], str]],
    vocabulary: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a binary Bag-of-Words feature matrix and integer label vector.

    Parameters
    ----------
    documents:
        List of (token_list, intent_tag) pairs produced by parse_intents().
    vocabulary:
        Sorted, deduplicated word list.  Column *i* of the output matrix
        corresponds to ``vocabulary[i]``.

    Returns
    -------
    X : np.ndarray  shape (n_docs, vocab_size)  dtype float32
        Binary BoW matrix.
    Y : np.ndarray  shape (n_docs,)             dtype int64
        Class label for each document.
    """
    intents_list: List[str] = list(dict.fromkeys(tag for _, tag in documents))
    vocab_map: Dict[str, int] = {word: idx for idx, word in enumerate(vocabulary)}
    n: int = len(documents)
    v: int = len(vocabulary)
    X = np.zeros((n, v), dtype=np.float32)
    Y = np.zeros(n, dtype=np.int64)
    for i, (token_list, tag) in enumerate(documents):
        for word in token_list:
            col = vocab_map.get(word)
            if col is not None:
                X[i, col] = 1.0
        Y[i] = intents_list.index(tag)
    return X, Y


def _build_tfidf_matrix(
    documents: List[Tuple[List[str], str]],
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Build a TF-IDF feature matrix using scikit-learn's TfidfVectorizer.

    Parameters
    ----------
    documents:
        List of (token_list, intent_tag) pairs.

    Returns
    -------
    X : np.ndarray  shape (n_docs, vocab_size)  dtype float32
    Y : np.ndarray  shape (n_docs,)             dtype int64
    vectorizer : fitted TfidfVectorizer  (needed to transform unseen sentences)
    """
    intents_list: List[str] = list(dict.fromkeys(tag for _, tag in documents))
    corpus: List[str] = [" ".join(tokens) for tokens, _ in documents]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus).toarray().astype(np.float32)
    Y = np.array(
        [intents_list.index(tag) for _, tag in documents], dtype=np.int64
    )
    return X, Y, vectorizer


def _build_gensim_matrix(
    documents: List[Tuple[List[str], str]],
) -> Tuple[np.ndarray, np.ndarray, Any, Any]:
    """
    Build a TF-IDF feature matrix using Gensim's Dictionary + TfidfModel.

    Requires gensim to be installed (``pip install gensim``).

    Parameters
    ----------
    documents:
        List of (token_list, intent_tag) pairs.

    Returns
    -------
    X          : np.ndarray  shape (n_docs, vocab_size)  dtype float32
    Y          : np.ndarray  shape (n_docs,)             dtype int64
    dictionary : gensim.corpora.Dictionary  (needed at inference time)
    tfidf      : gensim.models.TfidfModel   (needed at inference time)

    Raises
    ------
    ImportError
        When gensim is not installed.
    """
    if not _GENSIM_AVAILABLE:
        raise ImportError(
            "gensim is required for VectorizerMode.GENSIM. "
            "Install it with:  pip install gensim"
        )
    intents_list: List[str] = list(dict.fromkeys(tag for _, tag in documents))
    token_lists: List[List[str]] = [toks for toks, _ in documents]

    dictionary = corpora.Dictionary(token_lists)
    bow_corpus  = [dictionary.doc2bow(toks) for toks in token_lists]
    tfidf       = GensimTfidfModel(bow_corpus)

    n   = len(documents)
    v   = len(dictionary)
    X   = np.zeros((n, v), dtype=np.float32)
    for i, bow_vec in enumerate(bow_corpus):
        for token_id, score in tfidf[bow_vec]:
            X[i, token_id] = score

    Y = np.array(
        [intents_list.index(tag) for _, tag in documents], dtype=np.int64
    )
    return X, Y, dictionary, tfidf


def _build_feature_matrix(
    documents: List[Tuple[List[str], str]],
    vocabulary: List[str],
    mode: "VectorizerMode" = VectorizerMode.BOW,
) -> Tuple[np.ndarray, np.ndarray, Optional[Any], Optional[Any]]:
    """
    Unified feature-matrix builder.

    Returns
    -------
    X            : feature matrix  (n_docs, features)
    Y            : label vector    (n_docs,)
    vectorizer   : fitted sklearn TfidfVectorizer  — or None for BOW / Gensim
    gensim_state : (Dictionary, TfidfModel) tuple  — or None for BOW / TF-IDF
    """
    if mode == VectorizerMode.TFIDF:
        X, Y, vec = _build_tfidf_matrix(documents)
        return X, Y, vec, None

    if mode == VectorizerMode.GENSIM:
        X, Y, dictionary, tfidf = _build_gensim_matrix(documents)
        return X, Y, None, (dictionary, tfidf)

    # Default: BOW
    X, Y = _build_bow_matrix(documents, vocabulary)
    return X, Y, None, None


# ─────────────────────────────────────────────────────────────────────────────
# LLM wrapper — lazy-loads on first use
# ─────────────────────────────────────────────────────────────────────────────

class _PyAILLM:
    """Thin wrapper around the quantized model from pythonaibrain_llm."""

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._model_path: Optional[str] = None

    def _load(self) -> None:
        if not _PYAI_LLM_AVAILABLE:
            raise RuntimeError(
                "pythonaibrain-llm is not installed. "
                "Run: pip install pythonaibrain-llm"
            )
        try:
            validate_system()
        except Exception as exc:
            raise RuntimeError(f"System check failed: {exc}") from exc

        self._model_path = load_llm_model()
        logger.info("LLM model ready at: %s", self._model_path)

        try:
            from llama_cpp import Llama  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "It should ship automatically with pythonaibrain-llm."
            ) from exc

        llm_cfg = _cfg.llm
        self._model = Llama(
            model_path=self._model_path,
            n_ctx=llm_cfg.n_ctx,
            n_threads=llm_cfg.n_threads or os.cpu_count() or 4,
            verbose=llm_cfg.verbose,
        )
        logger.info("LLM loaded successfully.")

    def ask(self, prompt: Optional[str]) -> str:
        if self._model is None:
            self._load()
        clean = (prompt or "").strip()
        if not clean:
            return ""
        try:
            out = self._model(clean, max_tokens=_cfg.llm.max_tokens, stop=["</s>", "\n\n"], echo=False)
            return out["choices"][0]["text"].strip()
        except Exception as exc:
            logger.error("LLM inference error: %s", exc)
            return f"[LLM Error] {exc}"


_llm_instance: Optional[_PyAILLM] = None


def _get_llm() -> _PyAILLM:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = _PyAILLM()
    return _llm_instance


# ─────────────────────────────────────────────────────────────────────────────
# Weather helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_weather(city: str) -> Dict[str, Any]:
    if not _WEATHER_API_KEY:
        raise RuntimeError("weather_api_key not set in .env")
    params = urlencode({"q": city, "appid": _WEATHER_API_KEY, "units": "metric"})
    try:
        with urllib_request.urlopen(
            f"{_WEATHER_BASE_URL}?{params}", timeout=10
        ) as r:
            data: Dict[str, Any] = json.loads(r.read().decode())
    except URLError as exc:
        raise RuntimeError(f"Weather API request failed: {exc}") from exc
    if "weather" not in data:
        raise RuntimeError(
            f"Unexpected API response: {data.get('message', data)}"
        )
    return data


def get_weather(city: str) -> str:
    return _fetch_weather(city)["weather"][0]["main"]

def longitude(city: str) -> float:
    return _fetch_weather(city)["coord"]["lon"]

def latitude(city: str) -> float:
    return _fetch_weather(city)["coord"]["lat"]

def wind_speed(city: str) -> float:
    return _fetch_weather(city)["wind"]["speed"]


# ─────────────────────────────────────────────────────────────────────────────
# Help string & fallback intents
# ─────────────────────────────────────────────────────────────────────────────

Help: str = """# Pythonaibrain
PythonAIBrain is a versatile, plug-and-play Python package for building offline
intelligent AI assistants.  Visit https://pypi.org/project/pythonaibrain for
full documentation.
"""

intents_json: Dict[str, Any] = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "What's up?", "Howdy"],
            "responses": ["Hello! How can I help you today?", "Hey there!"],
        },
        {
            "tag": "bye",
            "patterns": ["Bye", "See you soon", "Take care"],
            "responses": ["Bye! Have a great day", "See you"],
        },
    ]
}


# ─────────────────────────────────────────────────────────────────────────────
# IntentsManager
# ─────────────────────────────────────────────────────────────────────────────

class IntentsManager:
    def __init__(self, intents_path: str = r".\intents.json") -> None:
        self.intents_path = intents_path
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if os.path.exists(self.intents_path):
            try:
                with open(self.intents_path, "r", encoding="utf-8") as fh:
                    self.data = json.load(fh)
                return
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("IntentsManager.load: %s", exc)
        try:
            with resources.open_text("pyaitk", "intents.json") as fh:
                self.data = json.load(fh)
        except Exception as exc:
            logger.error("Failed to load built-in intents: %s", exc)
            self.data = {"intents": []}

    def save(self) -> None:
        try:
            with open(self.intents_path, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=4)
        except OSError as exc:
            logger.error("IntentsManager.save: %s", exc)

    def add_intent(
        self, tag: str, patterns: List[str], responses: List[str]
    ) -> None:
        intents = self.data.setdefault("intents", [])
        for intent in intents:
            if intent["tag"] == tag:
                intent["responses"] = list(
                    set(intent.get("responses", [])) | set(responses)
                )
                for p in patterns:
                    if p not in intent.get("patterns", []):
                        intent.setdefault("patterns", []).append(p)
                self.save()
                return
        intents.append({"tag": tag, "patterns": patterns, "responses": responses})
        self.save()

    def add_search_intent(self, query: str, search_results: List[str]) -> None:
        tag = f"search_{query.strip().lower().replace(' ', '_')[:30]}"
        self.add_intent(tag, [query], search_results or ["Sorry, no results found."])


# ─────────────────────────────────────────────────────────────────────────────
# Frame classifier — singleton, trains once per process
# ─────────────────────────────────────────────────────────────────────────────

class FrameClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_FRAME_TRAIN: List[str] = [
    "How are you?", "Open the door", "The sun rises in the east",
    "What time is it?", "Close the window", "She is reading a book",
    "Is this your pen?", "Start the engine", "He likes football",
    "Where do you live?", "1+1 is 2", "I am Divyanshu",
    "Myself Divyanshu", "Do you know", "Shutdown /s /t 0", "Mkdir", "Start",
]
_FRAME_MAP: Dict[int, str] = {
    0: "Statement", 1: "Question", 2: "Command", 3: "Answer",
    4: "Name",      5: "Know",     6: "Shutdown", 7: "Make Dir", 8: "Start",
}
_FRAME_LABELS: List[int] = [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 3, 4, 4, 5, 6, 7, 8]

_frame_cache: Optional[Tuple[Any, Any, Dict[int, str]]] = None
# Stores (vectorizer_or_None, gensim_state_or_None, model, label_map)


def _get_frame_cache(
    mode: "VectorizerMode" = VectorizerMode.BOW,
) -> Tuple[Any, Any, Any, Dict[int, str]]:
    """
    Lazily build and cache the basic FrameClassifier.

    Parameters
    ----------
    mode : VectorizerMode
        Feature-extraction strategy.  Changing the mode after the first call
        resets the cache and retrains.

    Returns
    -------
    (sklearn_vec_or_None, gensim_state_or_None, model, label_map)
    """
    global _frame_cache
    # Re-train if mode changed or cache is cold
    if _frame_cache is None or _frame_cache[0] != mode.value:
        dummy_docs: List[Tuple[List[str], str]] = [
            (s.lower().split(), str(l))
            for s, l in zip(_FRAME_TRAIN, _FRAME_LABELS)
        ]
        vocab = sorted(set(w for toks, _ in dummy_docs for w in toks))
        X_np, _, vec, gensim_state = _build_feature_matrix(dummy_docs, vocab, mode)
        X = torch.tensor(X_np, dtype=torch.float32)
        y = torch.tensor(_FRAME_LABELS[:len(dummy_docs)], dtype=torch.long)
        model = FrameClassifier(X.shape[1], len(_FRAME_MAP))
        opt = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(150):
            model.train()
            loss = loss_fn(model(X), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        _frame_cache = (mode.value, vec, gensim_state, model, _FRAME_MAP)
        logger.info("FrameClassifier trained and cached (mode=%s).", mode.value)
    return _frame_cache[1], _frame_cache[2], _frame_cache[3], _frame_cache[4]


def _transform_sentence_to_tensor(
    sentence: str,
    mode: "VectorizerMode",
    vocabulary: List[str],
    vec: Optional[Any],
    gensim_state: Optional[Any],
) -> torch.Tensor:
    """
    Convert a raw sentence into a float32 feature tensor using the given mode.

    Parameters
    ----------
    sentence      : input text
    mode          : VectorizerMode (BOW / TFIDF / GENSIM)
    vocabulary    : word list used during training (needed for BOW)
    vec           : fitted TfidfVectorizer  (needed for TFIDF, else None)
    gensim_state  : (Dictionary, TfidfModel) tuple (needed for GENSIM, else None)
    """
    if mode == VectorizerMode.TFIDF:
        if vec is None:
            raise ValueError("TF-IDF vectorizer not fitted.")
        arr = vec.transform([sentence]).toarray().astype(np.float32)
        return torch.tensor(arr, dtype=torch.float32)

    if mode == VectorizerMode.GENSIM:
        if gensim_state is None or not _GENSIM_AVAILABLE:
            raise ValueError("Gensim state not available.")
        dictionary, tfidf_model = gensim_state
        tokens = sentence.lower().split()
        bow_vec = dictionary.doc2bow(tokens)
        tfidf_vec = tfidf_model[bow_vec]
        arr = np.zeros((1, len(dictionary)), dtype=np.float32)
        for token_id, score in tfidf_vec:
            arr[0, token_id] = score
        return torch.tensor(arr, dtype=torch.float32)

    # BOW
    lem = nltk.WordNetLemmatizer()
    words = [lem.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)]
    bow_row = [1.0 if w in words else 0.0 for w in vocabulary]
    return torch.tensor([bow_row], dtype=torch.float32)


def predict_frame(
    sentence: str,
    mode: "VectorizerMode" = VectorizerMode.BOW,
    vocabulary: Optional[List[str]] = None,
) -> str:
    vec, gensim_state, model, frame_map = _get_frame_cache(mode)
    t = _transform_sentence_to_tensor(
        sentence, mode, vocabulary or [], vec, gensim_state
    )
    with torch.no_grad():
        return frame_map.get(
            int(torch.argmax(model(t), dim=1).item()), "Statement"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Translation (seq-to-seq GRU) — singleton, trains once per process
# ─────────────────────────────────────────────────────────────────────────────

def _build_vocab(sentences: List[str]) -> Dict[str, int]:
    vocab: Dict[str, int] = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
    idx = 3
    for s in sentences:
        for w in s.lower().split():
            if w not in vocab:
                vocab[w] = idx
                idx += 1
    return vocab


def _encode_sentence(sentence: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(w, 0) for w in sentence.lower().split()]


def _pad_seq(seq: List[int], max_len: int) -> List[int]:
    return (seq + [0] * max_len)[:max_len]


class _TranslationDataset(Dataset):
    def __init__(
        self,
        corpus: List[Tuple[str, str]],
        src_vocab: Dict[str, int],
        tgt_vocab: Dict[str, int],
    ) -> None:
        self.pairs = corpus
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src = max(len(s.split()) for s, _ in corpus)
        self.max_tgt = max(len(t.split()) for _, t in corpus) + 2

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s, t = self.pairs[idx]
        return (
            torch.tensor(
                _pad_seq(_encode_sentence(s, self.src_vocab), self.max_src)
            ),
            torch.tensor(
                _pad_seq(
                    [1] + _encode_sentence(t, self.tgt_vocab) + [2],
                    self.max_tgt,
                )
            ),
        )


class _Encoder(nn.Module):
    def __init__(self, vs: int, ed: int, hd: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vs, ed)
        self.rnn = nn.GRU(ed, hd, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.rnn(self.embed(x))
        return h


class _Decoder(nn.Module):
    def __init__(self, vs: int, ed: int, hd: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vs, ed)
        self.rnn = nn.GRU(ed, hd, batch_first=True)
        self.fc = nn.Linear(hd, vs)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        o, h = self.rnn(self.embed(x.unsqueeze(1)), h)
        return self.fc(o.squeeze(1)), h


_translator_cache: Optional[Tuple] = None

_DEFAULT_CORPUS: List[Tuple[str, str]] = [
    ("mera naam ravi hai",  "my name is ravi"),
    ("tum kaise ho",        "how are you"),
    ("hola como estas",     "hello how are you"),
    ("je m'appelle pierre", "my name is pierre"),
    ("my name is ravi",     "my name is ravi"),
]


def _get_translator() -> Tuple:
    global _translator_cache
    if _translator_cache is not None:
        return _translator_cache
    sv = _build_vocab([s for s, _ in _DEFAULT_CORPUS])
    tv = _build_vocab([t for _, t in _DEFAULT_CORPUS])
    tiv = {v: k for k, v in tv.items()}
    ds = _TranslationDataset(_DEFAULT_CORPUS, sv, tv)
    ld = DataLoader(ds, batch_size=1, shuffle=True)
    enc = _Encoder(len(sv), 32, 64)
    dec = _Decoder(len(tv), 32, 64)
    eo = optim.Adam(enc.parameters(), lr=0.005)
    do = optim.Adam(dec.parameters(), lr=0.005)
    lf = nn.CrossEntropyLoss(ignore_index=0)
    for _ in range(100):
        for src, tgt in ld:
            eo.zero_grad()
            do.zero_grad()
            hid = enc(src)
            loss: torch.Tensor = torch.tensor(0.0)
            di = tgt[:, 0]
            for t in range(1, tgt.size(1)):
                out, hid = dec(di, hid)
                loss = loss + lf(out, tgt[:, t])
                di = tgt[:, t]
            loss.backward()
            eo.step()
            do.step()
    _translator_cache = (enc, dec, ds, sv, tv, tiv)
    logger.info("Translation model trained and cached.")
    return _translator_cache


def translate_to_en(message: str = "") -> str:
    enc, dec, ds, sv, _, tiv = _get_translator()
    enc.eval()
    dec.eval()
    src = _pad_seq(_encode_sentence(message, sv), ds.max_src)
    hid = enc(torch.tensor([src]))
    di = torch.tensor([1])
    result: List[str] = []
    with torch.no_grad():
        for _ in range(ds.max_tgt):
            out, hid = dec(di, hid)
            pred = out.argmax(1).item()
            if pred == 2:
                break
            result.append(tiv.get(int(pred), "?"))
            di = torch.tensor([pred])
    return " ".join(result)


def language_classifier(message: Optional[str] = None) -> str:
    if not message:
        return "english"
    words = set(message.lower().split())
    if words & {"hai", "hain", "mera", "naam", "tum", "kaise", "aap"}:
        return "hindi"
    if words & {"je", "tu", "vous", "est", "une", "mon", "ma"}:
        return "french"
    if words & {"hola", "como", "estas", "soy", "me", "es", "en"}:
        return "spanish"
    return "english"


# ─────────────────────────────────────────────────────────────────────────────
# Advanced frame classifier — separate singleton from the basic one
# ─────────────────────────────────────────────────────────────────────────────

class FrameClassifierAdvance(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_frame_advance_cache: Optional[Tuple] = None


def predictFrameAdvance(
    sentence: Optional[str] = None,
    mode: "VectorizerMode" = VectorizerMode.BOW,
) -> str:
    """
    Predict the frame (Statement / Question / Command / …) of *sentence*.

    Parameters
    ----------
    sentence : str | None
        Input text to classify.
    mode : VectorizerMode
        Feature-extraction backend.  Supports BOW, TFIDF, and GENSIM.
        Changing the mode after the first call resets and retrains the cache.
    """
    global _frame_advance_cache
    if _frame_advance_cache is None or _frame_advance_cache[0] != mode.value:
        dummy_docs: List[Tuple[List[str], str]] = [
            (s.lower().split(), str(l))
            for s, l in zip(_FRAME_TRAIN, _FRAME_LABELS)
        ]
        vocab = sorted(set(w for toks, _ in dummy_docs for w in toks))
        X_np, _, vec, gensim_state = _build_feature_matrix(dummy_docs, vocab, mode)
        X = torch.tensor(X_np, dtype=torch.float32)
        y = torch.tensor(_FRAME_LABELS[:len(dummy_docs)], dtype=torch.long)
        model = FrameClassifierAdvance(X.shape[1], len(_FRAME_MAP))
        opt = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(150):
            model.train()
            loss = loss_fn(model(X), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        _frame_advance_cache = (mode.value, vec, gensim_state, model, vocab, _FRAME_MAP)
        logger.info("FrameClassifierAdvance trained and cached (mode=%s).", mode.value)

    _, vec, gensim_state, model, vocab, fm = _frame_advance_cache
    t = _transform_sentence_to_tensor(sentence or "", mode, vocab, vec, gensim_state)
    with torch.no_grad():
        return fm.get(
            int(torch.argmax(model(t), dim=1).item()), "Statement"
        )

TRAIN_DATA = [
    ("Apple was founded by Steve Jobs in California.", {"entities": [
        (0,5,"ORG"),(21,31,"PERSON"),(35,45,"GPE")]}),
    ("Google is headquartered in Mountain View, California.", {"entities": [
        (0,6,"ORG"),(27,40,"GPE"),(42,52,"GPE")]}),
    ("Elon Musk leads Tesla and SpaceX.", {"entities": [
        (0,9,"PERSON"),(16,21,"ORG"),(26,32,"ORG")]}),
    ("Amazon was founded by Jeff Bezos in Seattle.", {"entities": [
        (0,6,"ORG"),(22,32,"PERSON"),(36,43,"GPE")]}),
    ("Microsoft is based in Redmond, Washington.", {"entities": [
        (0,9,"ORG"),(22,29,"GPE"),(31,41,"GPE")]}),
    ("Meta acquired Instagram for one billion dollars.", {"entities": [
        (0,4,"ORG"),(14,23,"ORG")]}),
    ("Netflix was co-founded by Reed Hastings in Los Gatos.", {"entities": [
        (0,7,"ORG"),(26,39,"PERSON"),(43,52,"GPE")]}),
    ("Satya Nadella became CEO of Microsoft in 2014.", {"entities": [
        (0,13,"PERSON"),(28,37,"ORG")]}),
    ("OpenAI released GPT-4 in March 2023.", {"entities": [
        (0,6,"ORG"),(16,21,"PRODUCT")]}),
    ("The United Nations was established in San Francisco in 1945.", {"entities": [
        (4,18,"ORG"),(38,51,"GPE")]}),
    ("Paris is the capital of France.", {"entities": [
        (0,5,"GPE"),(24,30,"GPE")]}),
    ("Angela Merkel led Germany for sixteen years.", {"entities": [
        (0,13,"PERSON"),(18,25,"GPE")]}),
    ("The Eiffel Tower is located in Paris, France.", {"entities": [
        (4,16,"LOC"),(31,36,"GPE"),(38,44,"GPE")]}),
    ("Tokyo hosted the Summer Olympics in 1964.", {"entities": [
        (0,5,"GPE"),(17,31,"EVENT")]}),
    ("The Amazon River flows through Brazil.", {"entities": [
        (4,16,"LOC"),(29,35,"GPE")]}),
    ("Mount Everest is the highest peak on Earth.", {"entities": [
        (0,13,"LOC"),(38,43,"LOC")]}),
    ("The Nile is the longest river in Africa.", {"entities": [
        (4,8,"LOC"),(33,39,"LOC")]}),
    ("Sydney is the largest city in Australia.", {"entities": [
        (0,6,"GPE"),(30,39,"GPE")]}),
    ("The Great Wall of China stretches thousands of miles.", {"entities": [
        (4,18,"LOC"),(22,27,"GPE")]}),
    ("Lionel Messi plays for Inter Miami in Major League Soccer.", {"entities": [
        (0,12,"PERSON"),(23,34,"ORG"),(38,57,"ORG")]}),
    ("Serena Williams won the Wimbledon championship multiple times.", {"entities": [
        (0,15,"PERSON"),(24,32,"EVENT")]}),
    ("The FIFA World Cup 2022 was held in Qatar.", {"entities": [
        (4,18,"EVENT"),(36,41,"GPE")]}),
    ("LeBron James plays for the Los Angeles Lakers.", {"entities": [
        (0,11,"PERSON"),(27,46,"ORG")]}),
    ("Roger Federer retired from professional tennis in 2022.", {"entities": [
        (0,13,"PERSON")]}),
    ("Manchester United signed Cristiano Ronaldo from Juventus.", {"entities": [
        (0,17,"ORG"),(25,42,"PERSON"),(48,56,"ORG")]}),
    ("Usain Bolt set the world record at the Berlin World Championships.", {"entities": [
        (0,10,"PERSON"),(39,45,"GPE"),(46,64,"EVENT")]}),
    ("Tiger Woods won the Masters Tournament in Augusta.", {"entities": [
        (0,11,"PERSON"),(20,38,"EVENT"),(42,49,"GPE")]}),
    ("The Boston Celtics won the NBA Finals in 2024.", {"entities": [
        (4,19,"ORG"),(27,37,"EVENT")]}),
    ("Novak Djokovic has won more Grand Slam titles than any other man.", {"entities": [
        (0,14,"PERSON"),(30,40,"EVENT")]}),
    ("Albert Einstein developed the theory of relativity.", {"entities": [
        (0,15,"PERSON")]}),
    ("NASA launched the James Webb Space Telescope in 2021.", {"entities": [
        (0,4,"ORG"),(17,44,"PRODUCT")]}),
    ("Marie Curie discovered polonium and radium.", {"entities": [
        (0,11,"PERSON"),(22,29,"PRODUCT"),(34,40,"PRODUCT")]}),
    ("SpaceX launched its Falcon 9 rocket from Cape Canaveral.", {"entities": [
        (0,6,"ORG"),(20,28,"PRODUCT"),(43,56,"LOC")]}),
    ("Tim Berners-Lee invented the World Wide Web in 1989.", {"entities": [
        (0,13,"PERSON"),(29,42,"PRODUCT")]}),
    ("The Large Hadron Collider is operated by CERN in Geneva.", {"entities": [
        (4,26,"PRODUCT"),(41,45,"ORG"),(49,55,"GPE")]}),
    ("DeepMind created AlphaGo, which defeated Lee Sedol.", {"entities": [
        (0,8,"ORG"),(17,24,"PRODUCT"),(41,51,"PERSON")]}),
    ("The Hubble Space Telescope was launched by NASA in 1990.", {"entities": [
        (4,26,"PRODUCT"),(43,47,"ORG")]}),
    ("William Shakespeare wrote Hamlet and Macbeth.", {"entities": [
        (0,18,"PERSON"),(25,31,"WORK_OF_ART"),(36,43,"WORK_OF_ART")]}),
    ("Beethoven composed his Ninth Symphony while deaf.", {"entities": [
        (0,9,"PERSON"),(23,39,"WORK_OF_ART")]}),
    ("J.K. Rowling created the Harry Potter series.", {"entities": [
        (0,11,"PERSON"),(24,44,"WORK_OF_ART")]}),
    ("Picasso painted Guernica in response to the Spanish Civil War.", {"entities": [
        (0,7,"PERSON"),(16,23,"WORK_OF_ART"),(44,61,"EVENT")]}),
    ("Gabriel Garcia Marquez wrote One Hundred Years of Solitude.", {"entities": [
        (0,22,"PERSON"),(29,58,"WORK_OF_ART")]}),
    ("Warren Buffett leads Berkshire Hathaway as chairman.", {"entities": [
        (0,14,"PERSON"),(21,40,"ORG")]}),
    ("The New York Stock Exchange was founded in 1792.", {"entities": [
        (4,28,"ORG")]}),
    ("Goldman Sachs reported record profits last quarter.", {"entities": [
        (0,12,"ORG")]}),
    ("JPMorgan Chase acquired Bear Stearns during the 2008 financial crisis.", {"entities": [
        (0,13,"ORG"),(24,36,"ORG")]}),
    ("The World Bank is headquartered in Washington.", {"entities": [
        (4,14,"ORG"),(35,45,"GPE")]}),
    ("Joe Biden was inaugurated as the 46th President of the United States.", {"entities": [
        (0,9,"PERSON"),(55,68,"GPE")]}),
    ("Vladimir Putin has led Russia since the early 2000s.", {"entities": [
        (0,13,"PERSON"),(22,28,"GPE")]}),
    ("Narendra Modi is the Prime Minister of India.", {"entities": [
        (0,12,"PERSON"),(39,44,"GPE")]}),
    ("Harvard University was founded in 1636 in Cambridge, Massachusetts.", {"entities": [
        (0,18,"ORG"),(42,51,"GPE"),(53,66,"GPE")]}),
    ("Oxford University awarded Stephen Hawking an honorary degree.", {"entities": [
        (0,16,"ORG"),(25,40,"PERSON")]}),
    ("Stanford University is in the heart of Silicon Valley.", {"entities": [
        (0,18,"ORG"),(40,54,"LOC")]}),
    ("The University of Tokyo is Japan's leading research institution.", {"entities": [
        (4,24,"ORG"),(28,33,"GPE")]}),
    ("The New York Times published an article by Thomas Friedman.", {"entities": [
        (4,18,"ORG"),(43,58,"PERSON")]}),
    ("CNN reported live from Kyiv during the conflict.", {"entities": [
        (0,3,"ORG"),(23,27,"GPE")]}),
    ("The Washington Post was acquired by Jeff Bezos in 2013.", {"entities": [
        (4,19,"ORG"),(36,46,"PERSON")]}),
    ("Spotify acquired the podcast network Gimlet Media.", {"entities": [
        (0,7,"ORG"),(37,49,"ORG")]}),
    ("The Mayo Clinic is one of the most respected hospitals in the United States.", {"entities": [
        (4,15,"ORG"),(62,75,"GPE")]}),
    ("Alexander Fleming discovered penicillin in 1928.", {"entities": [
        (0,17,"PERSON"),(29,39,"PRODUCT")]}),
    ("Nikola Tesla held over 300 patents during his lifetime.", {"entities": [
        (0,12,"PERSON")]}),
    ("Malala Yousafzai gave a speech at the United Nations in New York.", {"entities": [
        (0,16,"PERSON"),(37,51,"ORG"),(55,63,"GPE")]}),
    ("Frida Kahlo is celebrated for her self-portraits painted in Mexico.", {"entities": [
        (0,11,"PERSON"),(61,67,"GPE")]}),
    ("The headquarters of NATO is located in Brussels, Belgium.", {"entities": [
        (20,24,"ORG"),(40,47,"GPE"),(49,56,"GPE")]}),
    ("Cape Town is South Africa's second most populous city.", {"entities": [
        (0,9,"GPE"),(13,25,"GPE")]}),
    ("Oslo was ranked the most expensive city in Europe.", {"entities": [
        (0,4,"GPE"),(42,48,"LOC")]}),
    ("The Tesla Model S can travel over 400 miles on a single charge.", {"entities": [
        (4,18,"PRODUCT")]}),
    ("Apple released the AirPods Pro with active noise cancellation.", {"entities": [
        (0,5,"ORG"),(19,30,"PRODUCT")]}),
    ("Samsung unveiled the Galaxy S24 Ultra at its annual event.", {"entities": [
        (0,7,"ORG"),(21,38,"PRODUCT")]}),
    ("Boeing's 747 transformed long-haul air travel after its debut in 1969.", {"entities": [
        (0,6,"ORG"),(9,12,"PRODUCT")]}),
    ("The Moderna mRNA-1273 vaccine showed high efficacy in clinical trials.", {"entities": [
        (4,11,"ORG"),(12,22,"PRODUCT")]}),
    ("NASA's Voyager 1 is the most distant human-made object in space.", {"entities": [
        (0,4,"ORG"),(7,16,"PRODUCT")]}),
    ("Microsoft launched the Surface Pro as a laptop-tablet hybrid.", {"entities": [
        (0,9,"ORG"),(23,33,"PRODUCT")]}),
    ("The AstraZeneca Oxford vaccine was approved for emergency use in the UK.", {"entities": [
        (4,26,"PRODUCT"),(70,72,"GPE")]}),
    ("Google introduced the Pixel 8 camera with AI-enhanced night photography.", {"entities": [
        (0,6,"ORG"),(22,29,"PRODUCT")]}),
    ("SpaceX's Starship completed its first full orbital flight test in 2024.", {"entities": [
        (0,6,"ORG"),(9,17,"PRODUCT")]}),
    ("IBM's Deep Blue defeated Garry Kasparov in a chess match in 1997.", {"entities": [
        (0,3,"ORG"),(6,14,"PRODUCT"),(24,38,"PERSON"),(57,63,"GPE")]}),
    ("Neuralink implanted its N1 chip in the first human patient in 2024.", {"entities": [
        (0,8,"ORG"),(23,30,"PRODUCT")]}),
    ("Adobe released Firefly, an AI image-generation tool, in 2023.", {"entities": [
        (0,5,"ORG"),(15,22,"PRODUCT")]}),
    ("DJI's Phantom 4 drone became popular among filmmakers worldwide.", {"entities": [
        (0,3,"ORG"),(6,15,"PRODUCT")]}),
    ("The Raspberry Pi microcomputer was developed in Cambridge, England.", {"entities": [
        (4,16,"PRODUCT"),(53,61,"GPE"),(63,70,"GPE")]}),
    ("Sony's PlayStation 5 sold millions of units within days of launch.", {"entities": [
        (0,4,"ORG"),(7,20,"PRODUCT")]}),
    ("The Sahara Desert is the largest hot desert in the world.", {"entities": [
        (4,17,"LOC")]}),
    ("Victoria Falls straddles the border between Zambia and Zimbabwe.", {"entities": [
        (4,17,"LOC"),(43,49,"GPE"),(54,62,"GPE")]}),
    ("The Louvre Museum in Paris houses over 35,000 works of art.", {"entities": [
        (4,16,"LOC"),(20,25,"GPE")]}),
    ("Machu Picchu is an ancient Incan citadel set high in the Andes.", {"entities": [
        (0,11,"LOC"),(57,62,"LOC")]}),
    ("The Rocky Mountains stretch from Canada to New Mexico.", {"entities": [
        (4,19,"LOC"),(34,40,"GPE"),(44,54,"GPE")]}),
    ("Lake Baikal in Siberia holds about twenty percent of the world's fresh water.", {"entities": [
        (0,12,"LOC"),(16,23,"LOC")]}),
    ("The Grand Canyon attracts millions of tourists to Arizona each year.", {"entities": [
        (4,15,"LOC"),(57,64,"GPE")]}),
    ("The Colosseum in Rome is one of the most visited monuments in Europe.", {"entities": [
        (4,14,"LOC"),(18,22,"GPE"),(62,68,"LOC")]}),
    ("Angkor Wat in Cambodia is the world's largest religious monument.", {"entities": [
        (0,9,"LOC"),(13,21,"GPE")]}),
    ("The Mississippi River drains much of the central United States.", {"entities": [
        (4,21,"LOC"),(50,63,"GPE")]}),
    ("The Himalayas form a natural barrier between India and China.", {"entities": [
        (4,13,"LOC"),(44,49,"GPE"),(54,59,"GPE")]}),
    ("Death Valley in California holds the record for the highest air temperature.", {"entities": [
        (0,11,"LOC"),(15,25,"GPE")]}),
    ("The Serengeti National Park spans Tanzania and Kenya.", {"entities": [
        (4,27,"LOC"),(34,42,"GPE"),(47,52,"GPE")]}),
    ("Stonehenge is a prehistoric monument located on Salisbury Plain in England.", {"entities": [
        (0,10,"LOC"),(48,62,"LOC"),(66,73,"GPE")]}),
    ("The Mariana Trench is the deepest point in the Pacific Ocean.", {"entities": [
        (4,18,"LOC"),(48,61,"LOC")]}),
    ("Yellowstone National Park sits atop a massive volcanic hotspot in Wyoming.", {"entities": [
        (0,24,"LOC"),(68,75,"GPE")]}),
    ("The Rhine River winds through Switzerland, Germany, and the Netherlands.", {"entities": [
        (4,15,"LOC"),(29,40,"GPE"),(42,49,"GPE"),(59,72,"GPE")]}),
    ("Table Mountain overlooks Cape Town in South Africa.", {"entities": [
        (0,13,"LOC"),(24,33,"GPE"),(37,49,"GPE")]}),
    ("The Berlin Wall fell in November 1989, marking the end of the Cold War.", {"entities": [
        (4,15,"EVENT"),(62,69,"EVENT")]}),
    ("The Apollo 11 mission landed astronauts on the Moon in July 1969.", {"entities": [
        (4,14,"EVENT")]}),
    ("The French Revolution began in 1789 and reshaped Europe's political order.", {"entities": [
        (4,21,"EVENT")]}),
    ("The G20 Summit was held in New Delhi under India's presidency.", {"entities": [
        (4,14,"EVENT"),(27,36,"GPE"),(43,48,"GPE")]}),
    ("The Super Bowl LVIII was watched by over 100 million viewers.", {"entities": [
        (4,19,"EVENT")]}),
    ("World War II ended in Europe on 8 May 1945, known as Victory in Europe Day.", {"entities": [
        (0,11,"EVENT"),(66,76,"EVENT")]}),
    ("The Cannes Film Festival awarded the Palme d'Or to a French director.", {"entities": [
        (4,23,"EVENT"),(38,49,"AWARD")]}),
    ("The Boston Marathon was disrupted by bombings in April 2013.", {"entities": [
        (4,19,"EVENT")]}),
    ("The Paris Agreement was signed by nearly 200 countries at COP21.", {"entities": [
        (4,19,"LAW"),(57,62,"EVENT")]}),
    ("The Sundance Film Festival premiered the documentary in Park City.", {"entities": [
        (4,27,"EVENT"),(56,65,"GPE")]}),
    ("Hurricane Katrina devastated New Orleans in August 2005.", {"entities": [
        (0,17,"EVENT"),(29,40,"GPE")]}),
    ("The Chernobyl disaster in Ukraine triggered global nuclear safety reforms.", {"entities": [
        (4,20,"EVENT"),(24,31,"GPE")]}),
    ("The Arab Spring protests swept across Tunisia, Egypt, and Libya.", {"entities": [
        (4,15,"EVENT"),(38,44,"GPE"),(46,51,"GPE"),(57,62,"GPE")]}),
    ("The Tokyo 2020 Olympics were held in 2021 due to the pandemic.", {"entities": [
        (4,19,"EVENT")]}),
    ("The Nuremberg Trials held Nazi leaders accountable after World War II.", {"entities": [
        (4,21,"EVENT"),(58,69,"EVENT")]}),
    ("The Davos World Economic Forum convenes global leaders each January.", {"entities": [
        (4,28,"EVENT")]}),
    ("The Woodstock Music Festival drew hundreds of thousands to upstate New York.", {"entities": [
        (4,27,"EVENT"),(70,78,"GPE")]}),
    ("The Great Depression began with the Wall Street Crash of 1929.", {"entities": [
        (4,19,"EVENT"),(34,51,"EVENT")]}),
    ("The Srebrenica massacre was recognized as a genocide by the International Court of Justice.", {"entities": [
        (4,23,"EVENT"),(80,90,"ORG")]}),
    ("The Moon Landing in 1969 was broadcast live to millions around the world.", {"entities": [
        (4,15,"EVENT")]}),
    ("The Nobel Peace Prize was awarded to Malala Yousafzai in 2014.", {"entities": [
        (4,21,"AWARD"),(37,53,"PERSON")]}),
    ("The Pulitzer Prize for Fiction was won by Colson Whitehead.", {"entities": [
        (4,18,"AWARD"),(42,58,"PERSON")]}),
    ("The Grammy Award for Album of the Year went to Beyoncé.", {"entities": [
        (4,16,"AWARD"),(48,55,"PERSON")]}),
    ("The Booker Prize was awarded to Bernardine Evaristo for Girl, Woman, Other.", {"entities": [
        (4,17,"AWARD"),(32,50,"PERSON"),(55,74,"WORK_OF_ART")]}),
    ("Albert Camus received the Nobel Prize in Literature in 1957.", {"entities": [
        (0,11,"PERSON"),(25,49,"AWARD")]}),
    ("The BAFTA Award for Best Film went to Everything Everywhere All at Once.", {"entities": [
        (4,10,"AWARD"),(40,71,"WORK_OF_ART")]}),
    ("Simone Biles won the all-around gold medal at the World Artistic Gymnastics Championships.", {"entities": [
        (0,11,"PERSON"),(49,88,"EVENT")]}),
    ("The Academy Award for Best Picture was won by Parasite.", {"entities": [
        (4,18,"AWARD"),(47,55,"WORK_OF_ART")]}),
    ("The Fields Medal is awarded every four years to outstanding mathematicians under forty.", {"entities": [
        (4,16,"AWARD")]}),
    ("The Turner Prize celebrated contemporary British art for decades.", {"entities": [
        (4,16,"AWARD")]}),
    ("The Tony Award for Best Musical was given to Hamilton.", {"entities": [
        (4,14,"AWARD"),(43,51,"WORK_OF_ART")]}),
    ("The Golden Globe for Best Drama Series was awarded to Succession.", {"entities": [
        (4,16,"AWARD"),(55,65,"WORK_OF_ART")]}),
    ("Emma Thompson won the BAFTA for Best Actress for her role in Howards End.", {"entities": [
        (0,13,"PERSON"),(22,27,"AWARD"),(64,75,"WORK_OF_ART")]}),
    ("The Man Booker International Prize went to Han Kang for The Vegetarian.", {"entities": [
        (4,31,"AWARD"),(40,48,"PERSON"),(53,70,"WORK_OF_ART")]}),
    ("The César Award is France's most prestigious film honour.", {"entities": [
        (4,15,"AWARD"),(19,25,"GPE")]}),
    ("The Pritzker Architecture Prize was awarded to Renzo Piano in 1998.", {"entities": [
        (4,29,"AWARD"),(44,55,"PERSON")]}),
    ("Chimamanda Ngozi Adichie received the PEN Pinter Prize for her essays.", {"entities": [
        (0,24,"PERSON"),(37,52,"AWARD")]}),
    ("The Palme d'Or at Cannes was given to Parasite, directed by Bong Joon-ho.", {"entities": [
        (4,14,"AWARD"),(18,24,"EVENT"),(38,46,"WORK_OF_ART"),(60,73,"PERSON")]}),
    ("The Commonwealth Writers Prize recognised novels from across the globe.", {"entities": [
        (4,30,"AWARD")]}),
    ("The International Booker Prize was shared between author and translator.", {"entities": [
        (4,28,"AWARD")]}),
    ("The Turing Award is often called the Nobel Prize of computing.", {"entities": [
        (4,16,"AWARD"),(37,48,"AWARD")]}),
    ("Francis Ford Coppola won the Palme d'Or at Cannes for Apocalypse Now.", {"entities": [
        (0,19,"PERSON"),(29,39,"AWARD"),(43,49,"EVENT"),(54,69,"WORK_OF_ART")]}),
    ("The Anisfield-Wolf Book Award recognises works that confront racism.", {"entities": [
        (4,27,"AWARD")]}),
    ("The Sakharov Prize for Freedom of Thought is awarded by the European Parliament.", {"entities": [
        (4,19,"AWARD"),(64,82,"ORG")]}),
    ("The Europeans with Disabilities Act was signed into law by George H.W. Bush.", {"entities": [
        (4,35,"LAW"),(60,76,"PERSON")]}),
    ("The Universal Declaration of Human Rights was adopted by the United Nations in 1948.", {"entities": [
        (4,43,"LAW"),(58,72,"ORG")]}),
    ("The Clean Air Act was passed by the United States Congress in 1970.", {"entities": [
        (4,17,"LAW"),(40,53,"GPE")]}),
    ("The Paris Agreement commits countries to limiting global warming to 1.5 degrees.", {"entities": [
        (4,19,"LAW")]}),
    ("The Dodd-Frank Act was enacted in response to the 2008 financial crisis.", {"entities": [
        (4,18,"LAW")]}),
    ("The Children's Online Privacy Protection Act restricts data collection from minors.", {"entities": [
        (4,44,"LAW")]}),
    ("HIPAA protects patient health information in the United States.", {"entities": [
        (0,5,"LAW"),(47,60,"GPE")]}),
    ("The Sarbanes-Oxley Act was passed to improve corporate financial disclosures.", {"entities": [
        (4,23,"LAW")]}),
    ("The Kyoto Protocol set binding emissions targets for developed nations.", {"entities": [
        (4,19,"LAW")]}),
    ("The Geneva Conventions establish the standards of international humanitarian law.", {"entities": [
        (4,22,"LAW")]}),
    ("The Sherman Antitrust Act was used to break up Standard Oil in 1911.", {"entities": [
        (4,24,"LAW"),(53,65,"ORG")]}),
    ("The Voting Rights Act of 1965 was signed by President Lyndon B. Johnson.", {"entities": [
        (4,27,"LAW"),(55,72,"PERSON")]}),
    ("The Treaty of Versailles formally ended World War I in June 1919.", {"entities": [
        (4,24,"LAW"),(40,51,"EVENT")]}),
    ("The Patriot Act expanded surveillance powers of law enforcement after 9/11.", {"entities": [
        (4,15,"LAW")]}),
    ("The Digital Millennium Copyright Act regulates copyright in the internet age.", {"entities": [
        (4,37,"LAW")]}),
    ("The Magna Carta, signed in 1215, is considered the foundation of constitutional law.", {"entities": [
        (4,14,"LAW")]}),
    ("The Right to Information Act empowers citizens in India to access government records.", {"entities": [
        (4,31,"LAW"),(57,62,"GPE")]}),
    ("The Montreal Protocol was a landmark international agreement to protect the ozone layer.", {"entities": [
        (4,23,"LAW")]}),
    ("The Equal Pay Act prohibited wage discrimination based on sex in the United States.", {"entities": [
        (4,18,"LAW"),(77,90,"GPE")]}),
    ("The Basel III accord tightened capital requirements for banks worldwide.", {"entities": [
        (4,13,"LAW")]}),
    ("The African Union's Constitutive Act established the continent's governing body.", {"entities": [
        (4,35,"LAW"),(40,49,"ORG")]}),
    ("The Data Protection Act 2018 updated UK privacy law to align with the GDPR.", {"entities": [
        (4,27,"LAW"),(72,76,"LAW")]}),
    ("The Espionage Act has been used to prosecute government whistleblowers.", {"entities": [
        (4,17,"LAW")]}),
    ("Tuberculosis remains one of the leading infectious causes of death globally.", {"entities": [
        (0,13,"DISEASE")]}),
    ("The Ebola virus disease outbreak in West Africa in 2014 killed over 11,000 people.", {"entities": [
        (4,24,"DISEASE"),(34,45,"LOC")]}),
    ("Malaria is transmitted by Anopheles mosquitoes and kills hundreds of thousands annually.", {"entities": [
        (0,7,"DISEASE")]}),
    ("The 1918 Spanish Flu infected an estimated 500 million people worldwide.", {"entities": [
        (9,20,"DISEASE")]}),
    ("HIV/AIDS has claimed over 36 million lives since the epidemic began in the 1980s.", {"entities": [
        (0,8,"DISEASE")]}),
    ("Parkinson's disease causes tremors, stiffness, and difficulty with balance.", {"entities": [
        (0,19,"DISEASE")]}),
    ("Multiple sclerosis disrupts communication between the brain and body.", {"entities": [
        (0,19,"DISEASE")]}),
    ("Type 2 diabetes is closely linked to obesity and sedentary lifestyles.", {"entities": [
        (0,15,"DISEASE")]}),
    ("Researchers found a new treatment pathway for amyotrophic lateral sclerosis.", {"entities": [
        (47,75,"DISEASE")]}),
    ("Cholera outbreaks are common in areas lacking clean water and sanitation.", {"entities": [
        (0,7,"DISEASE")]}),
    ("Cystic fibrosis is caused by mutations in the CFTR gene.", {"entities": [
        (0,15,"DISEASE")]}),
    ("Dengue fever is endemic in over 100 countries and affects millions each year.", {"entities": [
        (0,12,"DISEASE")]}),
    ("The CDC issued a warning about the rising incidence of Lyme disease in the northeast.", {"entities": [
        (4,7,"ORG"),(56,68,"DISEASE")]}),
    ("Schizophrenia affects approximately one in one hundred people globally.", {"entities": [
        (0,13,"DISEASE")]}),
    ("Rheumatoid arthritis is an autoimmune disease that primarily affects the joints.", {"entities": [
        (0,21,"DISEASE")]}),
    ("Researchers at the University of Oxford developed a new treatment for sickle cell disease.", {"entities": [
        (19,39,"ORG"),(72,90,"DISEASE")]}),
    ("The SARS outbreak in 2003 spread rapidly through hospitals in Hong Kong.", {"entities": [
        (4,8,"DISEASE"),(60,69,"GPE")]}),
    ("Meningitis caused by Neisseria meningitidis can be life-threatening.", {"entities": [
        (0,10,"DISEASE")]}),
    ("Polio was declared eradicated in India in 2014 after years of vaccination campaigns.", {"entities":[
        (0,5,"DISEASE"),(36,41,"GPE")]}),
]

def predictNER(sentence: Optional[str] = "", train: Optional[bool] = False) -> Dict[str, Optional[str]]:
    if train:
        trainer = NERTrainer(TrainerConfig(n_iter=30))
        trainer.train(TRAIN_DATA, output_dir="models/my_ner")
    pipeline = NERPipeline.from_model_path("models/my_ner")
    result = pipeline.predict(sentence)

    return result.entities

# ─────────────────────────────────────────────────────────────────────────────
# Chatbot neural model
# ─────────────────────────────────────────────────────────────────────────────

class ChatbotModel(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


# ─────────────────────────────────────────────────────────────────────────────
# ChatbotAssistant — plain class, NOT @dataclass
# ─────────────────────────────────────────────────────────────────────────────

class ChatbotAssistant:
    """Low-level intent classifier and response dispatcher."""

    def __init__(
        self,
        intents_path: str,
        condition: bool = True,
        function_mapping: Optional[Dict[str, Any]] = None,
        vectorizer_mode: "VectorizerMode" = VectorizerMode.BOW,
    ) -> None:
        self.condition = condition
        self.model: Optional[nn.Module] = None
        self.intents_path = intents_path
        self.documents: List[Tuple[List[str], str]] = []
        self.vocabulary: List[str] = []
        self.intents: List[str] = []
        self.intents_responses: Dict[str, List[str]] = {}
        self.function_mapping: Dict[str, Any] = function_mapping or {}
        self.X: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        # Vectorizer state
        self.vectorizer_mode: VectorizerMode = vectorizer_mode
        self._sklearn_vec: Optional[Any] = None   # TfidfVectorizer (TFIDF mode)
        self._gensim_state: Optional[Any] = None  # (Dictionary, TfidfModel) (GENSIM mode)

    @staticmethod
    def tokenize_and_lemmatizer(text: str) -> List[str]:
        lem = nltk.WordNetLemmatizer()
        return [lem.lemmatize(w.lower()) for w in nltk.word_tokenize(text)]

    def bag_of_words(self, words: List[str]) -> List[int]:
        """Binary BoW vector (kept for backward compatibility)."""
        return [1 if w in words else 0 for w in self.vocabulary]

    def parse_intents(self) -> None:
        intents_data: Optional[Dict[str, Any]] = None
        if os.path.exists(self.intents_path):
            try:
                with open(self.intents_path, "r", encoding="utf-8") as fh:
                    intents_data = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("parse_intents: %s", exc)
        if intents_data is None:
            try:
                with resources.open_text("pyaitk", "intents.json") as fh:
                    intents_data = json.load(fh)
            except Exception as exc:
                logger.error("Cannot load intents: %s", exc)
                intents_data = intents_json
        for intent in intents_data["intents"]:
            if intent["tag"] not in self.intents:
                self.intents.append(intent["tag"])
                self.intents_responses[intent["tag"]] = intent["responses"]
            for pattern in intent.get("patterns", []):
                words = self.tokenize_and_lemmatizer(pattern)
                self.vocabulary.extend(words)
                self.documents.append((words, intent["tag"]))
        self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self) -> None:
        """Build feature matrix X and label array Y using the selected vectorizer mode."""
        self.X, self.Y, self._sklearn_vec, self._gensim_state = _build_feature_matrix(
            self.documents, self.vocabulary, self.vectorizer_mode
        )
        logger.info(
            "ChatbotAssistant: data prepared with mode=%s, X=%s",
            self.vectorizer_mode.value, self.X.shape,
        )

    def train_model(self, batch_size: int, lr: float, epochs: int) -> None:
        X_t = torch.tensor(self.X, dtype=torch.float32)
        Y_t = torch.tensor(self.Y, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X_t, Y_t), batch_size=batch_size, shuffle=True
        )
        self.model = ChatbotModel(self.X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        logger.info("Chatbot training started.")
        for epoch in range(epochs):
            running_loss = 0.0
            for bX, bY in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(bX), bY)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                logger.debug(
                    "Epoch %d/%d  loss=%.4f", epoch + 1, epochs, running_loss
                )
        logger.info("Chatbot training complete.")

    def save_model(self, model_path: str, dimension_path: str) -> None:
        import pickle
        torch.save(self.model.state_dict(), model_path)
        dim_data: Dict[str, Any] = {
            "input_size": int(self.X.shape[1]),
            "output_size": len(self.intents),
            "vocabulary": self.vocabulary,
            "intents": self.intents,
            "vectorizer_mode": self.vectorizer_mode.value,
        }
        with open(dimension_path, "w", encoding="utf-8") as fh:
            json.dump(dim_data, fh, indent=4)
        # Persist sklearn / Gensim state alongside the dimension file
        vec_path = dimension_path + ".vec.pkl"
        with open(vec_path, "wb") as fh:
            pickle.dump(
                {"sklearn_vec": self._sklearn_vec, "gensim_state": self._gensim_state},
                fh,
            )
        logger.info("ChatbotAssistant: model saved (mode=%s).", self.vectorizer_mode.value)

    def load_model(self, model_path: str, dimension_path: str) -> None:
        import pickle
        dim: Optional[Dict] = None
        try:
            with open(dimension_path, "r", encoding="utf-8") as fh:
                dim = json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass
        if dim is None:
            with resources.open_text("pyaitk", dimension_path) as fh:
                dim = json.load(fh)
        self.vocabulary = dim["vocabulary"]
        self.intents = dim["intents"]
        self.vectorizer_mode = VectorizerMode(dim.get("vectorizer_mode", VectorizerMode.BOW.value))
        # Restore sklearn / Gensim state
        vec_path = dimension_path + ".vec.pkl"
        if os.path.exists(vec_path):
            try:
                with open(vec_path, "rb") as fh:
                    vec_data = pickle.load(fh)
                self._sklearn_vec  = vec_data.get("sklearn_vec")
                self._gensim_state = vec_data.get("gensim_state")
            except Exception as exc:
                logger.warning("Could not load vectorizer state: %s", exc)
        self.model = ChatbotModel(dim["input_size"], dim["output_size"])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        try:
            self.model = torch.compile(self.model)  # type: ignore[assignment]
        except Exception as e:
            logger.warning(
                "torch.compile skipped (eager mode used): %s", e
            )
        logger.info("ChatbotAssistant: model loaded (mode=%s).", self.vectorizer_mode.value)

    def process_message(self, input_message: str) -> Any:
        if self.model is None:
            return "Model not loaded. Call load() or train() first."
        words = self.tokenize_and_lemmatizer(input_message)
        bag_tensor = _transform_sentence_to_tensor(
            input_message,
            self.vectorizer_mode,
            self.vocabulary,
            self._sklearn_vec,
            self._gensim_state,
        )
        self.model.eval()
        with torch.no_grad():
            preds = self.model(bag_tensor)
        predicted_intent: str = self.intents[
            int(torch.argmax(preds, dim=1).item())
        ]
        
        if predicted_intent == "fallback_search":
            try:
                s = Search(input_message)
                s.run()
                results = s.get_results_str()
                if self.condition:
                    IntentsManager(self.intents_path).add_search_intent(
                        input_message, results
                    )
                return results
            except Exception as exc:
                logger.error("Search: %s", exc)
                return f"[Error] {exc}"
        if predicted_intent == "help":
            return Help
        
        if predicted_intent in self.function_mapping:
            func = self.function_mapping[predicted_intent]
            try:
                result = func()
                return result if result is not None else f"Done: {predicted_intent}"
            except Exception as exc:
                logger.error("function_mapping[%s]: %s", predicted_intent, exc)
                return f"[Error in {predicted_intent}] {exc}"
        if self.intents_responses.get(predicted_intent):
            return random.choice(self.intents_responses[predicted_intent])
        return "I didn't understand that."


# ─────────────────────────────────────────────────────────────────────────────
# WebAssistant
# ─────────────────────────────────────────────────────────────────────────────

class WebAssistantModel(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WebAssistant:
    def __init__(
        self,
        intents_path: str,
        condition: bool = True,
        function_mapping: Optional[Dict[str, Any]] = None,
        vectorizer_mode: "VectorizerMode" = VectorizerMode.BOW,
    ) -> None:
        self.condition = condition
        self.model: Optional[nn.Module] = None
        self.intents_path = intents_path
        self.documents: List[Tuple[List[str], str]] = []
        self.vocabulary: List[str] = []
        self.intents: List[str] = []
        self.intents_responses: Dict[str, List[str]] = {}
        self.function_mapping: Dict[str, Any] = function_mapping or {}
        self.X: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.vectorizer_mode: VectorizerMode = vectorizer_mode
        self._sklearn_vec: Optional[Any] = None
        self._gensim_state: Optional[Any] = None

    @staticmethod
    def tokenize_and_lemmatizer(text: str) -> List[str]:
        lem = nltk.WordNetLemmatizer()
        return [lem.lemmatize(w.lower()) for w in nltk.word_tokenize(text)]

    def bag_of_words(self, words: List[str]) -> List[int]:
        """Binary BoW vector (kept for backward compatibility)."""
        return [1 if w in words else 0 for w in self.vocabulary]

    def parse_intents(self) -> None:
        intents_data: Optional[Dict[str, Any]] = None
        if os.path.exists(self.intents_path):
            try:
                with open(self.intents_path, "r", encoding="utf-8") as fh:
                    intents_data = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("WebAssistant.parse_intents: %s", exc)
        if intents_data is None:
            with resources.open_text("pyaitk", "intents.json") as fh:
                intents_data = json.load(fh)
        for intent in intents_data["intents"]:
            if intent["tag"] not in self.intents:
                self.intents.append(intent["tag"])
                self.intents_responses[intent["tag"]] = intent["responses"]
            for p in intent.get("patterns", []):
                words = self.tokenize_and_lemmatizer(p)
                self.vocabulary.extend(words)
                self.documents.append((words, intent["tag"]))
        self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self) -> None:
        self.X, self.Y, self._sklearn_vec, self._gensim_state = _build_feature_matrix(
            self.documents, self.vocabulary, self.vectorizer_mode
        )

    def train_model(self, batch_size: int, lr: float, epochs: int) -> None:
        X_t = torch.tensor(self.X, dtype=torch.float32)
        Y_t = torch.tensor(self.Y, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X_t, Y_t), batch_size=batch_size, shuffle=True
        )
        self.model = WebAssistantModel(self.X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        logger.info("WebAssistant training started.")
        for _ in range(epochs):
            for bX, bY in loader:
                optimizer.zero_grad()
                criterion(self.model(bX), bY).backward()
                optimizer.step()
        logger.info("WebAssistant training complete.")

    def save_model(self, model_path: str, dimension_path: str) -> None:
        import pickle
        torch.save(self.model.state_dict(), model_path)
        with open(dimension_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "input_size": int(self.X.shape[1]),
                    "output_size": len(self.intents),
                    "vocabulary": self.vocabulary,
                    "intents": self.intents,
                    "vectorizer_mode": self.vectorizer_mode.value,
                },
                fh,
            )
        vec_path = dimension_path + ".vec.pkl"
        with open(vec_path, "wb") as fh:
            pickle.dump(
                {"sklearn_vec": self._sklearn_vec, "gensim_state": self._gensim_state},
                fh,
            )

    def load_model(self, model_path: str, dimension_path: str) -> None:
        import pickle
        dim: Optional[Dict] = None
        try:
            with open(dimension_path, "r", encoding="utf-8") as fh:
                dim = json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass
        if dim is None:
            with resources.open_text("pyaitk", dimension_path) as fh:
                dim = json.load(fh)
        self.vocabulary = dim["vocabulary"]
        self.intents = dim["intents"]
        self.vectorizer_mode = VectorizerMode(dim.get("vectorizer_mode", VectorizerMode.BOW.value))
        vec_path = dimension_path + ".vec.pkl"
        if os.path.exists(vec_path):
            try:
                with open(vec_path, "rb") as fh:
                    vec_data = pickle.load(fh)
                self._sklearn_vec  = vec_data.get("sklearn_vec")
                self._gensim_state = vec_data.get("gensim_state")
            except Exception as exc:
                logger.warning("WebAssistant: could not load vectorizer state: %s", exc)
        self.model = WebAssistantModel(dim["input_size"], dim["output_size"])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, input_message: str) -> Optional[str]:
        if self.model is None:
            return None
        bag_tensor = _transform_sentence_to_tensor(
            input_message,
            self.vectorizer_mode,
            self.vocabulary,
            self._sklearn_vec,
            self._gensim_state,
        )
        self.model.eval()
        with torch.no_grad():
            idx = int(
                torch.argmax(
                    self.model(bag_tensor), dim=1
                ).item()
            )
        intent = self.intents[idx]
        return intent if self.intents_responses.get(intent) else None


# ─────────────────────────────────────────────────────────────────────────────
# OpenWebApps
# ─────────────────────────────────────────────────────────────────────────────

class OpenWebApps:
    def __init__(self, intents_path: Optional[str] = None) -> None:
        self.app_name: str = ""
        _path = intents_path if intents_path is not None else _cfg.webassistant.intents_path
        self.open_app = WebAssistant(_path)

    def load(self) -> bool:
        wc = _cfg.webassistant
        try:
            self.open_app.parse_intents()
            self.open_app.prepare_data()
            self.open_app.load_model(wc.model_path, wc.dimension_path)
            return True
        except Exception as exc:
            logger.error("OpenWebApps.load: %s", exc)
            return False

    def train(self) -> None:
        wc = _cfg.webassistant
        self.open_app.parse_intents()
        self.open_app.prepare_data()
        self.open_app.train_model(wc.batch_size, wc.learning_rate, wc.epochs)

    def save(self) -> bool:
        wc = _cfg.webassistant
        try:
            self.open_app.save_model(wc.model_path, wc.dimension_path)
            return True
        except Exception as exc:
            logger.error("OpenWebApps.save: %s", exc)
            return False

    def open(self, app_name: Optional[str] = "google") -> bool:
        try:
            webbrowser.open(self.app_name or app_name or "google")
            return True
        except Exception as exc:
            logger.error("OpenWebApps.open: %s", exc)
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Brain — plain class, NOT @dataclass
# ─────────────────────────────────────────────────────────────────────────────

class Brain:
    """
    Core AI Brain with intent classification, memory, and TTS support.

        with Brain() as brain:
            brain.load()
            print(brain.ask("Hello"))
    """

    __version__ = "1.1.9"
    __author__ = "Divyanshu Sinha"

    def __init__(
        self,
        intents_path: Optional[str] = None,
        condition: Optional[bool] = None,
        download: Optional[bool] = None,
        memory_path: Optional[str] = None,
        smart_memory: Optional[bool] = None,
        memory_fit_interval: Optional[int] = None,
        config: Optional["AppConfig"] = None,
        vectorizer_mode: "VectorizerMode" = VectorizerMode.BOW,
        **function_mapping: Any,
    ) -> None:
        # Resolve configuration: explicit kwargs → passed config → global config
        cfg = config or _cfg
        bc = cfg.brain
        _intents_path       = intents_path       if intents_path       is not None else bc.intents_path
        _condition          = condition          if condition          is not None else bc.condition
        _download           = download           if download           is not None else bc.download
        _memory_path        = memory_path        if memory_path        is not None else bc.memory_path
        _smart_memory       = smart_memory       if smart_memory       is not None else bc.smart_memory
        _memory_fit_interval = memory_fit_interval if memory_fit_interval is not None else bc.memory_fit_interval

        self.memory: Memory = build_memory(
            path=_memory_path,
            smart=_smart_memory,
            auto_load=cfg.memory.auto_load,
            auto_fit=_smart_memory and cfg.memory.auto_fit,
            fit_interval=_memory_fit_interval,
        )
        self.assistant = ChatbotAssistant(
            _intents_path, _condition,
            function_mapping=function_mapping,
            vectorizer_mode=vectorizer_mode,
        )
        self.vectorizer_mode: VectorizerMode = vectorizer_mode
        self.username: str = bc.username
        self._is_load = self._is_save = self._is_train = False
        self.no_of_query: int = 0
        self._config: "AppConfig" = cfg

    def __enter__(self) -> "Brain":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._is_train and not self._is_save:
            logger.warning("Brain exiting without saving – call brain.save().")

    def train(self) -> None:
        mc = self._config.model
        try:
            self.assistant.parse_intents()
            self.assistant.prepare_data()
            self.assistant.train_model(mc.batch_size, mc.learning_rate, mc.epochs)
            self._is_train = True
        except Exception as exc:
            logger.error("Brain.train: %s", exc)
            self._is_train = False

    def load(self) -> None:
        mc = self._config.model
        try:
            self.assistant.parse_intents()
            self.assistant.prepare_data()
            self.assistant.load_model(mc.model_path, mc.dimension_path)
            self._is_load = True
        except Exception as exc:
            logger.error("Brain.load: %s", exc)
            self._is_load = False

    def save(self) -> None:
        mc = self._config.model
        try:
            self.assistant.save_model(mc.model_path, mc.dimension_path)
            self._is_save = True
        except Exception as exc:
            logger.error("Brain.save: %s", exc)
            self._is_save = False

    def is_loaded(self) -> bool:  return self._is_load
    def is_saved(self) -> bool:   return self._is_save
    def is_trained(self) -> bool: return self._is_train
    def count_query(self) -> int: return self.no_of_query

    def translator(self, message: Optional[str] = None) -> str:
        return translate_to_en(message or "")

    def classify_language(self, message: Optional[str] = None) -> str:
        return language_classifier(message)

    def predict_message_type(self, message: Optional[str] = None) -> str:
        return predict_frame(
            message or "",
            mode=self.vectorizer_mode,
            vocabulary=self.assistant.vocabulary,
        )

    def predict_entitie(self, message: Optional[str] = None, train: Optional[bool] = False) -> Any:
        return predictNER(message or "", train)

    def pyai_say(self, *msg: Any, **kw: Any) -> None:
        print("PYAI :", *msg, **kw)

    def process_messages(
        self,
        message: Optional[str] = None,
        grammar: bool = True,
        TTS: bool = False,
    ) -> str:
        resp = str(self.assistant.process_message(message or ""))
        self.no_of_query += 1
        # SmartMemory handles persistence and auto-fit internally;
        # plain Memory still works identically — load/save is now a single
        # atomic call after remember() rather than a separate load-before.
        self.memory.remember(message, resp)
        self.memory.save_memory()
        out = GrammarCorrector().correct(resp) if grammar else resp
        if TTS:
            speak(out)
        return out

    def talk(
        self,
        message: Optional[str] = None,
        grammar: bool = False,
        TTS: bool = False,
    ) -> str:
        try:
            s = Search(message)
            s.run()
            result = s.get_results_str()
        except Exception:
            result = self.process_messages(message, grammar)
        if TTS:
            speak(result)
        return result

    def memorize_user_name(self, message: str = "") -> None:
        if self.predict_message_type(message) == "Name":
            self.memory.remember(
                self.username, str(self.predict_entitie(message))
            )
            self.memory.save_memory()

    def recall_user_name(self) -> str:
        return self.memory.recall(self.username)

    # ── SmartMemory convenience pass-throughs ─────────────────────────────────

    def search_memory(
        self, query: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over conversation history.
        Returns ranked list of {score, key, value, intent, cluster} dicts.
        Falls back to substring search when SmartMemory is unavailable.
        """
        if isinstance(self.memory, SmartMemory):
            return self.memory.semantic_search(query, top_k=top_k)
        # Plain Memory fallback — simple substring match
        needle = query.lower()
        results: List[Dict[str, Any]] = []
        for k, v in self.memory.items():
            if needle in k.lower() or needle in v.lower():
                results.append({"score": 1.0, "key": k, "value": v,
                                 "intent": "unknown", "cluster": -1})
                if len(results) >= top_k:
                    break
        return results

    def memory_intent(self, query: str) -> str:
        """Predict the intent of *query* from stored patterns."""
        if isinstance(self.memory, SmartMemory):
            return self.memory.predict_intent(query)
        return "unknown"

    def memory_report(self) -> Optional[Any]:
        """Return the MemorySummaryReport, or None if not fitted."""
        if isinstance(self.memory, SmartMemory):
            return self.memory.get_report()
        return None

    def export_memory_report(self, path: str = "memory_report.json") -> bool:
        """Export the cluster/intent analysis report as JSON."""
        if isinstance(self.memory, SmartMemory):
            return self.memory.export_report(path)
        logger.warning("export_memory_report: SmartMemory not active.")
        return False

    def fit_memory(self) -> bool:
        """Manually (re-)train the SmartMemory summarizer pipeline."""
        if isinstance(self.memory, SmartMemory):
            return self.memory.fit_summarizer(force=True)
        logger.warning("fit_memory: SmartMemory not active.")
        return False

    # ── message writing helpers ───────────────────────────────────────────────

    def _write_message(
        self, message: Optional[str] = None, TTS: bool = False
    ):
        if not message:
            return
        words = message.split()
        wi = 0
        for char in message:
            if TTS and char == " " and wi < len(words):
                speak(words[wi] + " ")
                wi += 1
            yield char

    def write(
        self, message: str = "Hi", set_timer: float = 0.05, TTS: bool = False
    ) -> None:
        from time import sleep
        for c in self._write_message(message, TTS):
            print(c, end="", flush=True)
            sleep(set_timer)
        print()

    def ask(self, query: Optional[str] = "Help", TTS: bool = False) -> str:
        ans = self.talk(message=query, grammar=True, TTS=TTS)
        self.write(ans)
        return ans


# ─────────────────────────────────────────────────────────────────────────────
# AdvanceBrain — plain class, NOT @dataclass
# ─────────────────────────────────────────────────────────────────────────────

class AdvanceBrain:
    """
    Advanced Brain that routes through the pythonaibrain-llm quantized LLM.

        with AdvanceBrain() as brain:
            brain.load()
            print(brain.process_messages("What is Python?"))
    """

    __version__ = "1.1.9"
    __author__ = "Divyanshu Sinha"

    def __init__(
        self,
        intents_path: Optional[str] = None,
        condition: Optional[bool] = None,
        config: Optional["AppConfig"] = None,
        vectorizer_mode: "VectorizerMode" = VectorizerMode.BOW,
        **function_mapping: Any,
    ) -> None:
        cfg = config or _cfg
        bc = cfg.brain
        _intents_path = intents_path if intents_path is not None else bc.intents_path
        _condition    = condition    if condition    is not None else bc.condition
        self.assistant = ChatbotAssistant(
            _intents_path,
            _condition,
            function_mapping=function_mapping,
            vectorizer_mode=vectorizer_mode,
        )
        self.vectorizer_mode: VectorizerMode = vectorizer_mode
        self.username: str = bc.username
        self._is_load = self._is_save = self._is_train = False
        self.no_of_query: int = 0
        self._config: "AppConfig" = cfg

    def __enter__(self) -> "AdvanceBrain":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._is_train and not self._is_save:
            logger.warning("AdvanceBrain exiting without saving trained model.")

    def train(self) -> None:
        mc = self._config.model
        try:
            self.assistant.parse_intents()
            self.assistant.prepare_data()
            self.assistant.train_model(mc.batch_size, mc.learning_rate, mc.epochs)
            self._is_train = True
        except Exception as exc:
            logger.error("AdvanceBrain.train: %s", exc)
            self._is_train = False

    def load(self) -> None:
        mc = self._config.model
        try:
            self.assistant.parse_intents()
            self.assistant.prepare_data()
            self.assistant.load_model(mc.model_path, mc.dimension_path)
            self._is_load = True
        except Exception as exc:
            logger.error("AdvanceBrain.load: %s", exc)
            self._is_load = False

    def save(self) -> None:
        mc = self._config.model
        try:
            self.assistant.save_model(mc.model_path, mc.dimension_path)
            self._is_save = True
        except Exception as exc:
            logger.error("AdvanceBrain.save: %s", exc)
            self._is_save = False

    def is_loaded(self) -> bool:  return self._is_load
    def is_saved(self) -> bool:   return self._is_save
    def is_trained(self) -> bool: return self._is_train
    def count_query(self) -> int: return self.no_of_query

    def translator(self, message: Optional[str] = None) -> str:
        return translate_to_en(message or "")

    def classify_language(self, message: Optional[str] = None) -> str:
        return language_classifier(message)

    def predict_message_type(self, message: Optional[str] = None) -> str:
        return predictFrameAdvance(message, mode=self.vectorizer_mode)

    def predict_entitie(self, message: Optional[str] = None, train: Optional[bool] = False) -> Any:
        return predictNER(message)

    def pyai_say(self, *msg: Any, **kw: Any) -> None:
        print("PYAI :", *msg, **kw)

    def process_messages(
        self,
        message: Optional[str] = None,
        grammar: bool = True,
        advance: bool = True,
        TTS: bool = False,
    ) -> str:
        self.no_of_query += 1
        raw = (
            _get_llm().ask(message)
            if advance
            else str(self.assistant.process_message(message or ""))
        )
        out = GrammarCorrector().correct(raw) if grammar else raw
        if TTS:
            speak(out)
        return out
