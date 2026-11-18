"""Helpers for optional dependencies.

Several parts of the specification name specialist libraries (LightGBM, arch, hmmlearn, cvxpy,
statsmodels, transformers). These are not required to run the core pipeline. Each module asks this
helper for the library; when it is missing, the module uses its own NumPy/SciPy implementation of
the same method, so results remain available on a minimal installation.
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

log = logging.getLogger("atlas")


def optional_import(name: str) -> ModuleType | None:
    """Return the imported module, or None if it is not installed."""
    try:
        return importlib.import_module(name)
    except Exception:  # ImportError and occasional binary-incompatibility errors
        return None


def require(name: str, extra: str) -> ModuleType:
    """Import a module that has no fallback, with an installation hint naming the pip extra."""
    module = optional_import(name)
    if module is None:
        raise ImportError(
            f"The '{name}' package is needed here. Install it with: pip install 'project-atlas[{extra}]'"
        )
    return module
