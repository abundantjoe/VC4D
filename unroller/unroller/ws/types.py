"""Shared contracts of the whole-scroll pipeline (docs/wholescroll/PLAN.md section 4.2).  FROZEN after WP0.

A change here is a WP0 amendment: a separate small commit on ``ws/base`` that every package branch rebases
onto.  ``tests/ws/test_contracts.py`` checks every signature below against the PLAN's section 4.2 text.

Conventions (PLAN section 4): coordinates are xyz in level-0 voxels; arrays are ZYX; NaN marks invalid
values; writes are atomic; results are deterministic.  The dataclasses coerce their arrays to the
documented dtypes on construction (``np.asarray``: no copy when the dtype already matches) and offer
``validate()``, which raises ``ValueError`` on inconsistent shapes.

Persistence: ``save_npz(path, obj)`` / ``load_npz(path, cls)`` for every dataclass here (and any other
dataclass whose fields are arrays, JSON-able values, dicts of those, or a ``Surface``): one ``.npz`` with
the arrays plus a JSON header carrying ``SCHEMA_VERSION`` and the class name; written to a temporary file
and renamed (atomic).  ``table_hash`` is the bit-exact identity of a table (S1 recompute check).
"""
from __future__ import annotations

import copy
import dataclasses
import enum
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

SCHEMA_VERSION = 1


class F(enum.IntFlag):            # uint32 per ridge point / node / cell
    PURE=1<<0; FUSED=1<<1; ONESIDED=1<<2; FOREIGN=1<<3; LOWCOH=1<<4; CORE=1<<5; RADIAL=1<<6; ORIENT_AMBIG=1<<7
    CRUSHED=1<<8; LOWMARGIN=1<<9; PRIOR=1<<10; UNCORROBORATED=1<<11; CONTESTED=1<<12; QUARANTINE=1<<13
    INTERP=1<<14; VIRTUAL=1<<15; SPLIT_CELL=1<<16; COLLAPSE=1<<17; FOLD=1<<18; ORDER=1<<19; DOUBLE_OCC=1<<20
    MISSING_LAYER=1<<21; FUSED_GEOM=1<<22; FUSED_UNRES=1<<23; SNAP_GUARD=1<<24; SNAP_CLAMP=1<<25; CHART_STEP=1<<26
    OUTER=1<<27; HOLDER=1<<28; ALIGN_BAD=1<<29; FIELD_RAMP=1<<30; EXTRAP=1<<31


class Status(enum.IntEnum):       # uint8 per node and per cell
    HOLE=0; CONFIRMED=1; SINGLE=2; QUARANTINE=3; PRIOR=4; CORE=5; INTERP=6; INSET=7; VIRTUAL=8


class Kind(enum.IntEnum):         # uint8 edge kind
    SAME=0; LADDER=1; L0COUNT=2; L0RECOUNT=3; PRIOR=4; HUMAN=5; LADDER2=6; W1_OVERLAP=10; W1_NORMAL=11


# per-vertex trace channels (TraceRec.channels): name -> dtype, each an (H, W) array on the trace grid
CHANNEL_DTYPES = {"H": np.float32, "width": np.float16, "contrast": np.float16, "conflict": np.float16,
                  "trunc": np.uint8, "plane_res": np.float16}


def _coerce(obj, spec):
    """Cast array fields of a dataclass instance in place: spec = {field: (dtype, trailing shape or None)}."""
    for name, (dt, _) in spec.items():
        v = getattr(obj, name)
        if v is not None:
            object.__setattr__(obj, name, np.asarray(v, dtype=dt))


def _check(obj, spec, n_name):
    """All fields in spec share the leading length of ``n_name``; trailing shapes match."""
    n = len(getattr(obj, n_name))
    for name, (_, tail) in spec.items():
        a = getattr(obj, name)
        if a.shape[:1] != (n,) or (tail is not None and a.shape[1:] != tail):
            raise ValueError(f"{type(obj).__name__}.{name}: shape {a.shape}, expected ({n},{'' if tail is None else tail})")


_RIDGE = {"idx": (np.int32, (3,)), "xyz": (np.float32, (3,)), "n": (np.float32, (3,)), "width": (np.float32, ()),
          "contrast": (np.float32, ()), "flags": (np.uint32, ())}
_SHEET_P = {"node": (np.int64, ()), "vth": (np.float64, ())}
_SHEET_N = {"ids": (np.int64, ()), "size": (np.int32, ()), "centroid": (np.float32, (3,)),
            "normal": (np.float32, (3,)), "flags": (np.uint32, ())}
_EDGES = {"i": (np.int64, ()), "j": (np.int64, ()), "d": (np.int32, ()), "count": (np.int32, ()),
          "w": (np.float32, ()), "kind": (np.uint8, ()), "qbin": (np.uint8, ()), "xyz": (np.float32, (3,))}
_SOL = {"ids": (np.int64, ()), "x": (np.int32, ()), "supernode": (np.int64, ()), "block": (np.int64, ()),
        "comp": (np.int32, ()), "conf": (np.float32, ()), "c1": (np.float32, ()), "cblk": (np.float32, ()),
        "corroborated": (bool, ()), "status": (np.uint8, ()), "flags": (np.uint32, ())}


@dataclass
class Ridge:                      # recomputed, never stored (only its hash)
    level: int; idx: np.ndarray        # (M,3) int32 ZYX at level
    xyz: np.ndarray                    # (M,3) float32 L0 (mean-pool centre mapping, d3/topo_proto.py:30)
    n: np.ndarray                      # (M,3) float32 oriented outer normal
    width: np.ndarray; contrast: np.ndarray   # (M,) float32, half-prominence width in level px
    flags: np.ndarray                  # (M,) uint32 F

    def __post_init__(self):
        self.level = int(self.level)
        _coerce(self, _RIDGE)

    def __len__(self):
        return len(self.idx)

    def validate(self) -> "Ridge":
        _check(self, _RIDGE, "idx")
        return self


@dataclass
class Sheetlets:
    node: np.ndarray                   # (M,) int64 node id per ridge point, -1 = none
    vth: np.ndarray                    # (M,) float64 local unwrapped angle, turns
    ids: np.ndarray; size: np.ndarray; centroid: np.ndarray; normal: np.ndarray; flags: np.ndarray  # per node

    def __post_init__(self):
        _coerce(self, _SHEET_P)
        _coerce(self, _SHEET_N)

    def index(self, node_ids) -> np.ndarray:
        """Row of each node id in the per-node arrays (``ids`` is sorted); -1 for ids not present."""
        node_ids = np.asarray(node_ids, np.int64)
        if len(self.ids) == 0:
            return np.full(node_ids.shape, -1, np.int64)
        k = np.clip(np.searchsorted(self.ids, node_ids), 0, len(self.ids) - 1)
        return np.where(self.ids[k] == node_ids, k, -1)

    def validate(self) -> "Sheetlets":
        _check(self, _SHEET_P, "node")
        _check(self, _SHEET_N, "ids")
        if len(self.ids) > 1 and not np.all(np.diff(self.ids) > 0):
            raise ValueError("Sheetlets.ids must be strictly increasing")
        m = self.node >= 0
        if m.any() and np.any(self.index(self.node[m]) < 0):
            raise ValueError("Sheetlets.node refers to ids that are not in Sheetlets.ids")
        if self.size.sum() != m.sum():
            raise ValueError("Sheetlets.size does not count the ridge points per node")
        return self


@dataclass
class Edges:
    i: np.ndarray; j: np.ndarray       # int64 node ids
    d: np.ndarray; count: np.ndarray   # int32
    w: np.ndarray                      # float32 (0 until calibrated)
    kind: np.ndarray; qbin: np.ndarray # uint8
    xyz: np.ndarray                    # (m,3) float32 location
    def concat(*es) -> "Edges":
        """``Edges.concat(a, b, ...)`` or ``a.concat(b, ...)``: all fields concatenated in order."""
        if len(es) == 1 and isinstance(es[0], (list, tuple)):
            es = tuple(es[0])
        if not es:
            return Edges.empty()
        return Edges(**{f: np.concatenate([np.asarray(getattr(e, f)).reshape((-1,) + _EDGES[f][1])
                                           for e in es]) for f in _EDGES})

    def __post_init__(self):
        _coerce(self, _EDGES)
        if self.xyz.size == 0:
            self.xyz = self.xyz.reshape(0, 3)

    @staticmethod
    def empty() -> "Edges":
        return Edges(**{f: np.zeros((0,) + tail, dt) for f, (dt, tail) in _EDGES.items()})

    def __len__(self):
        return len(self.i)

    def validate(self) -> "Edges":
        _check(self, _EDGES, "i")
        return self


@dataclass
class Solution:
    ids: np.ndarray; x: np.ndarray     # int64 / int32
    supernode: np.ndarray; block: np.ndarray; comp: np.ndarray       # int64 / int64 / int32
    conf: np.ndarray; c1: np.ndarray; cblk: np.ndarray               # float32
    corroborated: np.ndarray; status: np.ndarray; flags: np.ndarray  # bool / uint8 Status / uint32 F
    meta: dict    # chi, chi_ratio, energy, certificate(bool), stopped_by, iters, scale, quant_err, backend, wall_s

    def __post_init__(self):
        _coerce(self, _SOL)
        self.meta = dict(self.meta or {})

    @staticmethod
    def blank(ids, x=None, comp=None, certificate: bool = False, **meta) -> "Solution":
        """A Solution with every per-node array at its neutral value (supernode = block = id, conf = NaN,
        status HOLE, flags 0); ``x``/``comp`` default to 0."""
        ids = np.asarray(ids, np.int64)
        n = len(ids)
        z = lambda dt: np.zeros(n, dt)
        return Solution(ids=ids, x=z(np.int32) if x is None else x, supernode=ids.copy(), block=ids.copy(),
                        comp=z(np.int32) if comp is None else comp, conf=np.full(n, np.nan, np.float32),
                        c1=np.full(n, np.nan, np.float32), cblk=np.full(n, np.nan, np.float32),
                        corroborated=z(bool), status=z(np.uint8), flags=z(np.uint32),
                        meta=dict(meta, certificate=bool(certificate)))

    def copy(self) -> "Solution":
        return copy.deepcopy(self)

    def main_comp(self) -> int:
        """The MAIN component: ``meta['main_comp']`` if set, else the component with the most nodes
        (ties: the smallest id)."""
        if "main_comp" in self.meta:
            return int(self.meta["main_comp"])
        if len(self.comp) == 0:
            return 0
        u, c = np.unique(self.comp, return_counts=True)
        return int(u[np.argmax(c)])

    def __len__(self):
        return len(self.ids)

    def validate(self) -> "Solution":
        _check(self, _SOL, "ids")
        return self


def save_npz(path, obj) -> str:
    """Write a dataclass instance to ``path`` (``.npz`` appended if missing) atomically; returns the path.

    Arrays are stored as npz members; scalars, strings, lists and dicts as a JSON header (member
    ``__header__``) that also carries ``SCHEMA_VERSION`` and the class name.  Dict values and ``Surface``
    fields may themselves hold arrays."""
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("save_npz needs a dataclass instance")
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    arrays: dict[str, np.ndarray] = {}
    fields = {f.name: _encode(getattr(obj, f.name), f.name, arrays) for f in dataclasses.fields(obj)}
    header = {"schema_version": SCHEMA_VERSION, "class": type(obj).__name__, "fields": fields}
    arrays["__header__"] = np.frombuffer(json.dumps(header, default=_json_default).encode(), np.uint8)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)[:-4]}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def load_npz(path, cls):
    """Read what ``save_npz`` wrote; ``cls`` must be the class that was saved (checked by name), and the
    file's schema version must equal ``SCHEMA_VERSION``."""
    with np.load(str(path), allow_pickle=False) as z:
        header = json.loads(z["__header__"].tobytes().decode())
        if header.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema version {header.get('schema_version')} != {SCHEMA_VERSION}")
        if header.get("class") != cls.__name__:
            raise ValueError(f"{path}: holds a {header.get('class')}, not a {cls.__name__}")
        arrays = {k: z[k] for k in z.files if k != "__header__"}
    kw = {name: _decode(spec, arrays) for name, spec in header["fields"].items()}
    return cls(**kw)


def table_hash(*arrays) -> str:  # sha256 over (dtype, shape, C-order bytes)
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(a.dtype.str.encode())
        h.update(repr(tuple(int(s) for s in a.shape)).encode())
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def assign_status(sol: Solution, open_case_nodes: np.ndarray, w1_agree: np.ndarray, w3_agree: np.ndarray,
                  tau_m: float, r_core_mask: np.ndarray) -> np.ndarray:
    """The PLAN section 1 status rule, reference version (WP5 extends it).  Returns uint8 ``Status`` per node
    of ``sol``, in this order of precedence:

    * CORE        ``r_core_mask`` (r < r_core);
    * QUARANTINE  an open case after automatic escalation (``open_case_nodes``: bool mask per node, or the
                  node ids);
    * PRIOR       placed by prior only: flag ``F.PRIOR``, or not in the MAIN component (``sol.main_comp()``);
    * CONFIRMED   certified solve (``meta['certificate']``), conf >= tau_m, corroborated, no open case, no
                  witness disagreement, and at least one positive witness (W1 or W3);
    * SINGLE      the same, with no witness coverage at all;
    * QUARANTINE  everything else (a witness disagrees, low margin, uncorroborated or an uncertified solve,
                  without a case that would have closed it): nothing else counts as correct.

    ``w1_agree`` / ``w3_agree`` per node: > 0 agrees, < 0 disagrees, 0 or NaN = no coverage (bool arrays:
    True = agrees, False = no coverage)."""
    n = len(sol.ids)

    def per_node(a, name):
        a = np.asarray(a)
        if a.shape != (n,):
            raise ValueError(f"{name} must have one entry per node ({n}), got {a.shape}")
        return a

    core = per_node(r_core_mask, "r_core_mask").astype(bool)
    oc = np.asarray(open_case_nodes)
    if oc.dtype == bool:
        open_case = per_node(oc, "open_case_nodes")
    else:
        open_case = np.isin(sol.ids, oc.astype(np.int64))
    wit = []
    for a, name in ((w1_agree, "w1_agree"), (w3_agree, "w3_agree")):
        a = per_node(a, name).astype(np.float64)
        wit.append(np.where(np.isnan(a), 0.0, np.sign(a)))
    w1, w3 = wit
    positive = (w1 > 0) | (w3 > 0)
    contrary = (w1 < 0) | (w3 < 0)
    prior = ((sol.flags & np.uint32(F.PRIOR)) != 0) | (sol.comp != sol.main_comp())
    good = (bool(sol.meta.get("certificate", False)) & (sol.conf >= tau_m) & sol.corroborated & ~open_case
            & ~contrary)
    st = np.full(n, int(Status.QUARANTINE), np.uint8)
    st[good & positive] = Status.CONFIRMED
    st[good & ~positive] = Status.SINGLE
    st[prior] = Status.PRIOR
    st[open_case] = Status.QUARANTINE
    st[core] = Status.CORE
    return st


@runtime_checkable
class BandQuery(Protocol):
    def W(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...   # (W or NaN, Status uint8) at L0 xyz (N,3)


@dataclass
class TraceRec:
    surface: "Surface"; wrap_H0: float; seed: np.ndarray; pid: int
    channels: dict   # 'H' f32, 'width' f16, 'contrast' f16, 'conflict' f16, 'trunc' u1, 'plane_res' f16  (H,W) each
    meta: dict

    def __post_init__(self):
        self.wrap_H0 = float(self.wrap_H0)
        self.seed = np.asarray(self.seed, np.float64).reshape(3)
        self.pid = int(self.pid)
        self.channels = {k: np.asarray(v, CHANNEL_DTYPES.get(k, None)) for k, v in dict(self.channels or {}).items()}
        self.meta = dict(self.meta or {})

    def validate(self) -> "TraceRec":
        shp = self.surface.shape
        miss = set(CHANNEL_DTYPES) - set(self.channels)
        if miss:
            raise ValueError(f"TraceRec.channels lacks {sorted(miss)}")
        for k, v in self.channels.items():
            if v.shape != shp:
                raise ValueError(f"TraceRec.channels[{k!r}] has shape {v.shape}, surface grid is {shp}")
            if k in CHANNEL_DTYPES and v.dtype != CHANNEL_DTYPES[k]:
                raise ValueError(f"TraceRec.channels[{k!r}] has dtype {v.dtype}, expected {np.dtype(CHANNEL_DTYPES[k])}")
        return self


# ------------------------------------------------------------------ npz encoding helpers
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, enum.Enum):
        return o.value
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _encode(v, key, arrays):
    from ..surface import Surface

    if isinstance(v, np.ndarray):
        arrays[key] = v
        return {"array": key}
    if isinstance(v, Surface):
        arrays[key + ".points"] = v.points
        return {"surface": key + ".points", "scale": list(v.scale), "meta": json.loads(json.dumps(v.meta, default=_json_default))}
    if isinstance(v, dict):
        if not all(isinstance(k, str) for k in v):
            raise TypeError(f"{key}: dict keys must be strings")
        return {"dict": {k: _encode(x, f"{key}.{k}", arrays) for k, x in v.items()}}
    if isinstance(v, (np.generic,)):
        v = v.item()
    if isinstance(v, enum.Enum):
        v = v.value
    return {"json": json.loads(json.dumps(v, default=_json_default))}


def _decode(spec, arrays):
    from ..surface import Surface

    if "array" in spec:
        return arrays[spec["array"]]
    if "surface" in spec:
        return Surface(arrays[spec["surface"]], tuple(spec["scale"]), spec["meta"])
    if "dict" in spec:
        return {k: _decode(s, arrays) for k, s in spec["dict"].items()}
    return spec["json"]
