"""
Memory.py  –  Production-grade Memory Module for PythonAIBrain
==============================================================

Provides two public classes:

    Memory
        Classic key/value episodic memory used by Brain & AdvanceBrain
        (load_memory / remember / recall / save_memory contract that
        core.py already calls).

    SmartMemory
        Drop-in superset of Memory that also integrates the
        MemorySummarizer pipeline so that every saved pattern is
        automatically analysed, clustered, and retrievable through
        semantic search — not just exact-key lookup.

Architecture
------------
    ┌──────────────────────────────────────────────────────┐
    │  SmartMemory                                         │
    │   ├─ _store : dict[str, str]  (raw episodic store)   │
    │   ├─ _summarizer : MemorySummarizer  (ML pipeline)   │
    │   ├─ _dirty : bool  (lazy re-fit guard)              │
    │   └─ _lock : threading.RLock  (thread-safe ops)      │
    └──────────────────────────────────────────────────────┘

Usage in core.py  (drop-in replacement):

    # Plain usage — identical to old Memory
    from .Memory import Memory
    self.memory = Memory(path="memory.json")

    # Smart usage — same API + semantic search
    from .Memory import SmartMemory
    self.memory = SmartMemory(path="memory.json", auto_fit=True)

    # SmartMemory-only extras
    result = self.memory.semantic_search("tell me a joke")
    self.memory.fit_summarizer()
    report  = self.memory.get_report()
    self.memory.export_report("memory_report.json")
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from .config import get_config

logger = logging.getLogger(__name__)

# ── optional summarizer import (graceful degradation) ─────────────────────────
try:
    from SummarizerAI import MemorySummarizer          # type: ignore[import]
    _SUMMARIZER_AVAILABLE = True
except:
    try:
        # Allow import when running from inside the package
        from .SummarizerAI import MemorySummarizer     # type: ignore[import]
        _SUMMARIZER_AVAILABLE = True
    except ImportError:
        _SUMMARIZER_AVAILABLE = False
        logger.warning(
            "summarizer.py not found – SmartMemory will fall back to "
            "plain Memory (no ML clustering / semantic search)."
        )

__all__ = ["Memory", "SmartMemory", "MemoryEntry"]

# ─────────────────────────────────────────────────────────────────────────────
# Build Config
# ─────────────────────────────────────────────────────────────────────────────

def build_config(path: str | None= "./config.pbcfg"):
    config = get_config(path)
    summarizer = config.summarizer

    return {
        "latent_dim": summarizer.latent_dim,
        "hidden_dim": summarizer.hidden_dim,
        "ae_epochs": summarizer.ae_epochs,
        "ae_lr": summarizer.ae_lr,
        "ae_batch_size": summarizer.ae_batch_size,
        "device": "cpu",
        "embedding": {
            "max_features": 3000,
            "ngram_range": (1, 3),
            "sublinear_tf": True,
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single episodic memory record."""
    key: str
    value: str
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MemoryEntry":
        return MemoryEntry(
            key=d["key"],
            value=d["value"],
            timestamp=d.get("timestamp", 0.0),
            access_count=d.get("access_count", 0),
            last_accessed=d.get("last_accessed", 0.0),
        )


def _atomic_write(path: Path, text: str) -> None:
    """Write to a temp file then rename for atomic replacement."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
# Memory  –  Classic episodic store
# ─────────────────────────────────────────────────────────────────────────────

class Memory:
    """
    Thread-safe, JSON-backed key/value memory store.

    Fully compatible with the existing core.py contract:
        memory.load_memory()
        memory.remember(key, value)
        memory.recall(key)
        memory.save_memory()

    Extended API:
        memory.forget(key)            → delete one entry
        memory.wipe()                 → clear all entries (in-memory)
        memory.keys()                 → iterator over stored keys
        memory.items()                → iterator over (key, value) pairs
        memory.to_dict()              → {key: value} snapshot
        memory.size                   → number of entries
        memory.path                   → resolved Path to backing file
    """

    _FORMAT_VERSION = 2

    def __init__(
        self,
        path: str = "memory.json",
        max_entries: int = 10_000,
        auto_load: bool = True,
    ) -> None:
        self._path = Path(path).resolve()
        self._max = max_entries
        self._lock = threading.RLock()

        # OrderedDict so LRU eviction stays O(1)
        self._entries: OrderedDict[str, MemoryEntry] = OrderedDict()

        if auto_load and self._path.exists():
            self.load_memory()

    # ── public properties ─────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    # ── core.py contract ──────────────────────────────────────────────────────

    def load_memory(self) -> None:
        """Load (or reload) memory from disk.  Silent no-op if file absent."""
        if not self._path.exists():
            logger.debug("Memory file not found; starting empty: %s", self._path)
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load memory from %s: %s", self._path, exc)
            return

        with self._lock:
            self._entries.clear()
            if isinstance(raw, dict) and raw.get("__version__") == self._FORMAT_VERSION:
                # New rich format
                for rec in raw.get("entries", []):
                    try:
                        e = MemoryEntry.from_dict(rec)
                        self._entries[e.key] = e
                    except (KeyError, TypeError):
                        pass
            elif isinstance(raw, dict):
                # Legacy flat {key: value} format – migrate transparently
                for k, v in raw.items():
                    self._entries[k] = MemoryEntry(key=k, value=str(v))
            else:
                logger.warning("Unrecognised memory file format; ignoring.")

        logger.info("Memory loaded: %d entries from %s", len(self._entries), self._path)

    def remember(self, key: Optional[str], value: Optional[str]) -> None:
        """Store or update a key/value pair.  Evicts LRU entry when full."""
        if key is None:
            return
        key = str(key).strip()
        value = str(value) if value is not None else ""

        with self._lock:
            if key in self._entries:
                entry = self._entries[key]
                entry.value = value
                entry.touch()
                # Move to end (most-recently-used)
                self._entries.move_to_end(key)
            else:
                if len(self._entries) >= self._max:
                    evicted_key, _ = self._entries.popitem(last=False)
                    logger.debug("Memory full – evicted LRU key: %r", evicted_key)
                self._entries[key] = MemoryEntry(key=key, value=value)

        logger.debug("Remembered: %r → %r", key, value[:60] if len(value) > 60 else value)

    def recall(self, key: Optional[str], default: str = "") -> str:
        """Retrieve the value for *key*; returns *default* if not found."""
        if key is None:
            return default
        key = str(key).strip()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return default
            entry.touch()
            self._entries.move_to_end(key)
            return entry.value

    def save_memory(self) -> None:
        """Persist current memory state to disk atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload: Dict[str, Any] = {
                "__version__": self._FORMAT_VERSION,
                "saved_at": time.time(),
                "entry_count": len(self._entries),
                "entries": [e.to_dict() for e in self._entries.values()],
            }
        try:
            _atomic_write(self._path, json.dumps(payload, indent=2, ensure_ascii=False))
        except OSError as exc:
            logger.error("Failed to save memory to %s: %s", self._path, exc)
            return
        logger.info("Memory saved: %d entries → %s", len(self._entries), self._path)

    # ── extended API ──────────────────────────────────────────────────────────

    def forget(self, key: str) -> bool:
        """Remove *key* from memory.  Returns True if the key existed."""
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                logger.debug("Forgot key: %r", key)
                return True
        return False

    def wipe(self) -> None:
        """Clear all in-memory entries (does not touch disk until save_memory)."""
        with self._lock:
            self._entries.clear()
        logger.info("Memory wiped (in-memory only).")

    def keys(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._entries.keys()))

    def items(self) -> Iterator[Tuple[str, str]]:
        with self._lock:
            return iter([(k, e.value) for k, e in self._entries.items()])

    def to_dict(self) -> Dict[str, str]:
        """Return a plain {key: value} snapshot (no metadata)."""
        with self._lock:
            return {k: e.value for k, e in self._entries.items()}

    def __len__(self) -> int:
        return self.size

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return str(key) in self._entries

    def __repr__(self) -> str:
        return f"<Memory path={self._path!r} entries={self.size}>"


# ─────────────────────────────────────────────────────────────────────────────
# SmartMemory  –  Memory + MemorySummarizer integration
# ─────────────────────────────────────────────────────────────────────────────

class SmartMemory(Memory):
    """
    Drop-in superset of Memory that transparently integrates the
    MemorySummarizer ML pipeline (TF-IDF → Autoencoder → Clustering →
    IntentClassifier → PatternMatcher).

    When *auto_fit* is True (default) the summarizer is (re-)fitted
    automatically after every *fit_interval* calls to remember().  This
    keeps semantic search up-to-date without blocking the caller.

    Extra public API (no-ops if summarizer unavailable):
        fit_summarizer()              → train / retrain the ML pipeline
        semantic_search(text, top_k) → ranked [(score, key, value)]
        predict_intent(text)         → str intent label
        get_report()                 → MemorySummaryReport | None
        export_report(path)          → write JSON report to disk

    Thread safety
        All public methods acquire _lock.  The background fit thread also
        holds the lock only while reading the store snapshot; training
        happens outside the lock so memory reads/writes are never blocked.
    """

    def __init__(
        self,
        path: str = "memory.json",
        max_entries: int = 10_000,
        auto_load: bool = True,
        auto_fit: bool = True,
        fit_interval: int = 20,
        summarizer_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(path=path, max_entries=max_entries, auto_load=auto_load)

        self._auto_fit = auto_fit and _SUMMARIZER_AVAILABLE
        self._fit_interval = max(1, fit_interval)
        self._summarizer_config = summarizer_config or build_config()

        self._summarizer: Optional[Any] = None   # MemorySummarizer | None
        self._dirty = True                        # needs (re-)fit
        self._remember_count = 0
        self._fit_thread: Optional[threading.Thread] = None
        self._fit_lock = threading.Lock()         # guards fit itself

        if not _SUMMARIZER_AVAILABLE:
            logger.warning(
                "SmartMemory: summarizer unavailable – running as plain Memory."
            )

    # ── overridden remember ───────────────────────────────────────────────────

    def remember(self, key: Optional[str], value: Optional[str]) -> None:
        super().remember(key, value)
        with self._lock:
            self._dirty = True
            self._remember_count += 1
            should_fit = (
                self._auto_fit
                and self._remember_count % self._fit_interval == 0
            )
        if should_fit:
            self._fit_background()

    # ── ML API ───────────────────────────────────────────────────────────────

    def fit_summarizer(self, force: bool = False) -> bool:
        """
        (Re-)train the MemorySummarizer on the current memory store.
        Blocks the caller until training is complete.
        Returns True on success, False if unavailable or store is empty.
        """
        if not _SUMMARIZER_AVAILABLE:
            logger.warning("fit_summarizer: summarizer module not available.")
            return False

        with self._lock:
            snapshot = self.to_dict()

        if len(snapshot) < 2:
            logger.warning("fit_summarizer: need ≥2 entries to train (got %d).", len(snapshot))
            return False

        with self._fit_lock:
            try:
                logger.info("SmartMemory: fitting summarizer on %d patterns…", len(snapshot))
                s = MemorySummarizer(config=self._summarizer_config)
                s.fit(snapshot)
                with self._lock:
                    self._summarizer = s
                    self._dirty = False
                logger.info("SmartMemory: summarizer fit complete.")
                return True
            except Exception as exc:
                logger.error("SmartMemory: summarizer fit failed: %s", exc)
                return False

    def semantic_search(
        self,
        text: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Semantic (cosine-similarity) search over memory.

        Returns a list of dicts, each with:
            score   : float  [0, 1]
            key     : str    (original input key)
            value   : str    (stored response)
            intent  : str    (predicted intent)
            cluster : int    (cluster assignment)
        """
        if not self._ensure_summarizer():
            # Graceful degradation: substring fallback
            return self._substring_search(text, top_k)

        with self._lock:
            s = self._summarizer

        try:
            result = s.query(text)
            out = []
            for m in result.get("top_matches", [])[:top_k]:
                out.append({
                    "score":   m["score"],
                    "key":     m["input"],
                    "value":   m["response"],
                    "intent":  result.get("predicted_intent", "unknown"),
                    "cluster": result.get("cluster_id", -1),
                })
            return out
        except Exception as exc:
            logger.error("semantic_search failed: %s", exc)
            return self._substring_search(text, top_k)

    def predict_intent(self, text: str) -> str:
        """
        Predict the intent of *text* using the trained IntentClassifier.
        Falls back to 'unknown' if the summarizer is unavailable.
        """
        if not self._ensure_summarizer():
            return "unknown"
        with self._lock:
            s = self._summarizer
        try:
            result = s.query(text)
            return result.get("predicted_intent", "unknown")
        except Exception as exc:
            logger.error("predict_intent failed: %s", exc)
            return "unknown"

    def get_report(self) -> Optional[Any]:
        """Return the MemorySummaryReport dataclass, or None."""
        with self._lock:
            return self._summarizer.report_ if self._summarizer else None

    def export_report(self, path: str = "memory_report.json") -> bool:
        """
        Write the MemorySummaryReport to *path* as JSON.
        Returns True on success.
        """
        if not self._ensure_summarizer():
            logger.warning("export_report: summarizer not ready.")
            return False
        with self._lock:
            s = self._summarizer
        try:
            s.export_report(path)
            return True
        except Exception as exc:
            logger.error("export_report failed: %s", exc)
            return False

    def print_report(self) -> None:
        """Pretty-print the cluster report to stdout."""
        if not self._ensure_summarizer():
            print("[SmartMemory] No report – call fit_summarizer() first.")
            return
        with self._lock:
            s = self._summarizer
        s.print_report()

    def is_summarizer_fitted(self) -> bool:
        with self._lock:
            return self._summarizer is not None and not self._dirty

    # ── internal helpers ──────────────────────────────────────────────────────

    def _ensure_summarizer(self) -> bool:
        """Fit the summarizer if needed; return True if available."""
        if not _SUMMARIZER_AVAILABLE:
            return False
        with self._lock:
            needs_fit = self._summarizer is None or self._dirty
        if needs_fit:
            return self.fit_summarizer()
        return True

    def _fit_background(self) -> None:
        """Kick off a non-blocking background fit (best-effort)."""
        if self._fit_lock.locked():
            # A fit is already in progress
            return

        def _worker():
            self.fit_summarizer()

        t = threading.Thread(target=_worker, daemon=True, name="SmartMemory-fit")
        self._fit_thread = t
        t.start()

    def _substring_search(
        self, text: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """Simple case-insensitive substring fallback when ML is unavailable."""
        needle = text.lower()
        results = []
        with self._lock:
            for k, e in self._entries.items():
                if needle in k.lower() or needle in e.value.lower():
                    results.append({
                        "score":   1.0,
                        "key":     k,
                        "value":   e.value,
                        "intent":  "unknown",
                        "cluster": -1,
                    })
                    if len(results) >= top_k:
                        break
        return results

    def __repr__(self) -> str:
        fitted = "fitted" if (self._summarizer and not self._dirty) else "unfitted"
        return (
            f"<SmartMemory path={self._path!r} entries={self.size} "
            f"summarizer={fitted}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def build_memory(
    path: str = "memory.json",
    smart: bool = True,
    **kwargs: Any,
) -> Memory:
    """
    Factory that returns a SmartMemory when *smart* is True and the
    summarizer is importable; otherwise returns a plain Memory.

    Example
    -------
        from Memory import build_memory
        mem = build_memory("memory.json", smart=True, fit_interval=50)
    """
    if smart and _SUMMARIZER_AVAILABLE:
        return SmartMemory(path=path, **kwargs)
    return Memory(path=path, **{k: v for k, v in kwargs.items()
                                if k in ("max_entries", "auto_load")})