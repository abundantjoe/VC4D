"""Whole-scroll unrolling pipeline (``docs/wholescroll/PLAN.md``).

WP0 foundation: ``types`` (shared contracts, FROZEN), ``frame`` (umbilicus and polar frame, FROZEN),
``fixtures`` (test beds with exact truth; frozen API), ``channels`` (post-hoc per-vertex trace channels),
``manifest`` (stage manifest, parameter hash, resume, atomic writes) and ``_l1ref`` (verbatim copy of the
exact L1 min-cut prototype ``exp/l1_mincut.py``, reference only).  Submodules are imported on demand.
"""
from .types import SCHEMA_VERSION

__version__ = "0.1.0"

__all__ = ["SCHEMA_VERSION", "__version__"]
