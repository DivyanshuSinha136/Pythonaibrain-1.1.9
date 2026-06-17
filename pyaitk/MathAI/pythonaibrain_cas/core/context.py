"""
core/context.py
===============
CAS session context: controls output format, precision, assumptions.
"""

from __future__ import annotations
from typing import Dict, Optional


class CASContext:
    """
    Thread-local CAS configuration.

    Attributes:
        precision   : decimal digits for evalf
        output_mode : 'str' | 'latex' | 'mathml'
        simplify    : auto-simplify on construction
        assumptions : global symbol assumptions
    """

    _defaults = {
        'precision': 15,
        'output_mode': 'str',
        'auto_simplify': False,
        'domain': 'complex',  # 'real' or 'complex'
    }

    def __init__(self, **kw):
        self._cfg: Dict = dict(self._defaults)
        self._cfg.update(kw)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._cfg[name]
        except KeyError:
            raise AttributeError(f"Unknown CAS setting: {name!r}")

    def set(self, **kw) -> "CASContext":
        """Return new context with overrides."""
        new = CASContext(**self._cfg)
        new._cfg.update(kw)
        return new

    def __repr__(self):
        return f"CASContext({self._cfg})"


# Default global context
default_context = CASContext()
