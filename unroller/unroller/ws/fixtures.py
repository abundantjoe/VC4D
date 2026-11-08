"""Test beds with exact truth, and truth-built stand-ins for every stage's input (PLAN section 4.2, WP0).

They make the work packages independent: WP2 can test edges on ``truth_ridge`` before WP1 exists, WP4/WP6
can use ``truth_sheetlets`` with the exact integer x, WP7 the exact band, WP8 ``truth_traces``.  The API
(the functions of PLAN section 4.2) is FROZEN; new tiers or variants are registered, not edited in
(``VARIANTS``; WP12's ``ws/testbed.py`` registers its variants on import).

Tiers (``testbed(name, seed, variant, levels)``):
  d3small      the D3 prototype phantom (``Phantom`` of d3/band_trace.py, seed 3: 128 x 256^2, 8 turns,
               pitch 13, fused contacts), CT + exact H / D grids from the cache ``ph_small_s3.npz``
               (env ``UNROLLER_WS_D3SMALL``, else ``<cache>/d3small_s<seed>.npz``, else generated: ~3 min).
               ``seed=0`` (the default) means the dev seed 3 (PLAN 7.1: "d3small seed 3").
  ws-small     ``WholeScroll("small", seed)``  128 x 256^2, 8 wraps (unroller/wholescroll.py)
  ws-small_x2  ``WholeScroll((128, 512, 512), seed, wraps=8)``: pitch about 26-28 vx at L0 (L0/L1 hand-off)
  ws-medium    ``WholeScroll("medium", seed)`` 256 x 640^2, 20 wraps
  ws-large     ``WholeScroll("large", seed)``  512 x 1536^2, 40 wraps, tile-wise volume (``levels=1`` only)
  ws-tiny      ``WholeScroll((48, 128, 128), seed, wraps=3)``: test-only, for fast contract tests
WholeScroll tiers are cached (pickled phantom + L0 volume) in ``<cache>`` = env ``UNROLLER_WS_CACHE`` or
``~/.cache/unroller/ws``, keyed by tier, seed, variant and a hash of the generator sources.

Truth (``Truth``): H = t / 2 pi of the nearest sheet point (t = the spiral parameter, so H increases with
chi * theta; chi = +1 for these phantoms), d = signed distance to the sheet centre (+ outward), (s, z) =
the exact unrolled position, ink(s, z).  For d3small, H and d are trilinear samples of the cached exact
grids (like d3/band_trace.py's ``HV`` / ``DV``, so the D3 numbers reproduce); elsewhere they are exact.

Units: L0 voxels, xyz; ``Ridge.idx`` ZYX at the ridge's level.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass

import numpy as np

from ..surface import Surface
from ..volume import ArrayVolume, Volume
from .frame import TWO_PI, Frame, wrap
from .types import CHANNEL_DTYPES, F, BandQuery, Ridge, Sheetlets, Solution, Status, TraceRec

# d3/band_trace.py:27-31 (the phantom every D3 number was measured on), seed 3
D3SMALL_PARAMS = dict(size=(128, 256, 256), r0=14.0, pitch=13.0, turns=8.0,
                      gap_model=dict(thick_median=5.0, thick_sigma_ln=0.3, gap_median=7.0, gap_sigma_ln=0.9,
                                     corr_vx=60.0),
                      two_layer=True, thickness=5.0, warp_amp=3.0, warp_scale=48.0, tear_fraction=0.02,
                      psf_sigma=0.8, clip_background=True, noise=8.0, ellipticity=0.15, delamination=1.0,
                      delam_fraction=0.15)
D3SMALL_SEED = 3
WS_TIERS = {"ws-small": dict(size="small"), "ws-small_x2": dict(size=(128, 512, 512), wraps=8),
            "ws-medium": dict(size="medium"), "ws-large": dict(size="large"),
            "ws-tiny": dict(size=(48, 128, 128), wraps=3)}
TRACE_RADIUS_STEPS = 24      # truth_traces: patch half-size in grid steps (24 generations, as band_trace.py)
N_SECTORS = 8                # truth_sheetlets: angular sectors per turn (nodes span < 1/8 turn)
SMOOTH_SIGMA_VX = 8.0        # planted 'smooth': gaussian edge of the block (D3 topo_smooth16: sigma 8 vx)
FIXTURE_VERSION = 1          # bump to invalidate caches when this module changes how it builds a tier


def cache_dir() -> str:
    d = os.environ.get("UNROLLER_WS_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "unroller", "ws")
    os.makedirs(d, exist_ok=True)
    return d


def _src_hash() -> str:
    """Hash of the sources that generate a phantom (cache key)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha256(str(FIXTURE_VERSION).encode())
    for f in ("phantom.py", "wholescroll.py", "sector_fit.py", "realism.py", "_kernels.py"):
        p = os.path.join(here, f)
        if os.path.exists(p):
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:12]


# ------------------------------------------------------------------------------------------- truth
@dataclass
class Truth:
    src: object                      # the Phantom / WholeScroll: exact geometry
    shape: tuple                     # L0 volume shape (Z, Y, X)
    frame: Frame | None = None       # the true frame (Frame.from_truth)
    name: str = ""
    seed: int = 0
    variant: str = "base"
    H_grid: np.ndarray | None = None  # d3small: exact H / d on the L0 voxel grid (sampled trilinearly)
    D_grid: np.ndarray | None = None

    def _grid(self, g, p):
        """Trilinear sample of an L0 grid; NaN outside the grid box (no extrapolation)."""
        p = np.asarray(p, np.float64)
        out = ArrayVolume(g).sample(p).astype(np.float64)
        hi = np.array(g.shape[::-1], np.float64) - 1.0
        out[~np.all((p >= 0) & (p <= hi), -1)] = np.nan
        return out

    def _locate(self, p):
        p = np.asarray(p, np.float64)
        return self.src._locate(p)

    def H(self, p) -> np.ndarray:        # t/2pi of the nearest sheet point (chirality +1 by construction)
        if self.H_grid is not None:
            return self._grid(self.H_grid, p)
        s, _, _ = self._locate(p)
        return self.H_of_s(s)

    def d(self, p) -> np.ndarray:        # signed distance to the sheet centre
        if self.D_grid is not None:
            return self._grid(self.D_grid, p)
        return np.asarray(self._locate(p)[2], np.float64)

    def s_z(self, p) -> np.ndarray:      # exact unrolled (s, z)
        s, z, _ = self._locate(p)
        return np.stack([s, z], -1)

    def ink(self, s, z) -> np.ndarray:
        return np.asarray(self.src.ink_at(s, z), np.float64)

    # helpers (not part of the frozen contract, stable nonetheless)
    @property
    def length(self) -> float:
        return float(self.src.length)

    def t_of_s(self, s):
        return np.interp(np.asarray(s, np.float64), self.src._sc, self.src._tc)

    def s_of_t(self, t):
        return np.interp(np.asarray(t, np.float64), self.src._tc, self.src._sc)

    def H_of_s(self, s):
        s = np.asarray(s, np.float64)
        out = self.t_of_s(s) / TWO_PI
        return np.where(np.isfinite(s), out, np.nan)

    def point(self, s, z, offset=0.0):
        """World xyz of material (s, z) (z = the material axis coordinate), ``offset`` along the outward normal."""
        return self.src.point(s, z, offset)

    def present(self, s, z):
        """Papyrus present (not torn or missing) at material (s, z)."""
        s = np.asarray(s, np.float64)
        z = np.asarray(z, np.float64)
        s, z = np.broadcast_arrays(s, z)
        return (self.src._img(self.src.keep, s, z) > 0.5) & (s > 0) & (s < self.src.length)

    def thickness(self, s):
        return np.asarray(self.src.local_thickness(np.asarray(s, np.float64)), np.float64)


# ------------------------------------------------------------------------------------------- test beds
def _base_variant(tier, tier_kw, seed):
    from ..wholescroll import WholeScroll

    return WholeScroll(seed=seed, **tier_kw)


VARIANTS = {"base": _base_variant}   # variant -> fn(tier name, tier kwargs, seed) -> WholeScroll


def _cached_wholescroll(name, seed, variant, need_volume):
    from ..wholescroll import _ChunkCache

    if variant not in VARIANTS:
        try:
            from . import testbed as _wp12  # noqa: F401  (WP12 registers its variants on import)
        except ImportError:
            pass
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (registered: {sorted(VARIANTS)})")
    base = os.path.join(cache_dir(), f"{name}_s{seed}_{variant}_{_src_hash()}")
    src = None
    if os.path.exists(base + ".pkl"):
        try:
            with open(base + ".pkl", "rb") as f:
                src = pickle.load(f)
        except Exception:  # a truncated or stale cache: rebuild
            src = None
    if src is None:
        src = VARIANTS[variant](name, WS_TIERS[name], seed)
        cache = src._cache
        src._cache = None
        tmp = f"{base}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(src, f, protocol=5)
        os.replace(tmp, base + ".pkl")
        src._cache = cache
    if getattr(src, "_cache", None) is None:
        src._cache = _ChunkCache(int(0.5 * 2**30), None)
    vol = None
    if need_volume:
        if os.path.exists(base + "_vol.npz"):
            with np.load(base + "_vol.npz") as z:
                vol = z["vol"]
        else:
            vol = src.volume()
            tmp = f"{base}_vol.{os.getpid()}.tmp.npz"
            np.savez_compressed(tmp, vol=vol)
            os.replace(tmp, base + "_vol.npz")
    return src, vol


def _d3small_phantom(seed):
    from ..phantom import Phantom

    return Phantom(seed=seed, **D3SMALL_PARAMS)


def _d3small_arrays(ph, seed):
    """(vol, H, D) of d3small: from the cache, else generated exactly as d3/band_trace.py:36-46."""
    env = os.environ.get("UNROLLER_WS_D3SMALL")
    path = os.path.join(cache_dir(), f"d3small_s{seed}.npz")
    for p in ([env] if (env and seed == D3SMALL_SEED) else []) + [path]:
        if p and os.path.exists(p):
            with np.load(p) as z:
                vol, H, D = z["vol"], z["H"], z["D"]
            _check_d3small(ph, H, D, p)
            return vol, H, D
    vol = ph.volume()
    Z, Y, X = vol.shape
    H = np.empty(vol.shape, np.float32)
    D = np.empty(vol.shape, np.float32)
    yy, xx = np.mgrid[0:Y, 0:X].astype(float)
    for zi in range(Z):
        w = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, float(zi))], 1)
        s, _, d = ph._locate(w)
        H[zi] = (np.interp(s, ph._sc, ph._tc) / (2 * np.pi)).reshape(Y, X)
        D[zi] = d.reshape(Y, X)
    tmp = f"{path[:-4]}.{os.getpid()}.tmp.npz"
    np.savez(tmp, vol=vol, H=H, D=D)
    os.replace(tmp, path)
    return vol, H, D


def _check_d3small(ph, H, D, path, n=256):
    """The cached grids must be what the current phantom code computes (checked at n fixed voxels)."""
    rng = np.random.default_rng(12345)
    Z, Y, X = H.shape
    zyx = np.stack([rng.integers(0, Z, n), rng.integers(0, Y, n), rng.integers(0, X, n)], 1)
    s, _, d = ph._locate(zyx[:, ::-1].astype(float))
    h = (np.interp(s, ph._sc, ph._tc) / (2 * np.pi)).astype(np.float32)
    got_h, got_d = H[tuple(zyx.T)], D[tuple(zyx.T)]
    if not (np.array_equal(h, got_h, equal_nan=True) and np.array_equal(d.astype(np.float32), got_d, equal_nan=True)):
        raise RuntimeError(f"{path} does not match the current phantom code; delete it to regenerate")


def testbed(name: str, seed: int = 0, variant: str = "base", levels: int = 2) -> tuple["Volume", Truth, Frame]:
    """'d3small' (d3/ph_small_s3.npz, cached), 'ws-small', 'ws-small_x2', 'ws-medium', 'ws-large' (tile-wise).

    Returns (volume with ``levels`` mean-pooled levels, Truth, the true Frame).  See the module docstring
    for the tiers, the cache and ``ws-tiny``."""
    if name == "d3small":
        if variant != "base":
            raise ValueError("d3small has only the 'base' variant")
        seed = D3SMALL_SEED if seed == 0 else int(seed)
        ph = _d3small_phantom(seed)
        vol, H, D = _d3small_arrays(ph, seed)
        frame = Frame.from_truth(ph)
        truth = Truth(src=ph, shape=tuple(vol.shape), frame=frame, name=name, seed=seed, variant=variant,
                      H_grid=H, D_grid=D)
        return ArrayVolume(vol, levels=levels), truth, frame
    if name not in WS_TIERS:
        raise ValueError(f"unknown test bed {name!r} (d3small, {', '.join(WS_TIERS)})")
    tile_wise = name == "ws-large"
    if tile_wise and levels != 1:
        raise NotImplementedError("ws-large is tile-wise with one level here; its 3-level OME-Zarr "
                                  "materialisation is WP12's (call testbed('ws-large', levels=1))")
    src, vol = _cached_wholescroll(name, int(seed), variant, need_volume=not tile_wise)
    frame = Frame.from_truth(src)
    truth = Truth(src=src, shape=tuple(src.size), frame=frame, name=name, seed=int(seed), variant=variant)
    V = src.volume_obj() if tile_wise else ArrayVolume(vol, levels=levels, voxel_size=7.91)
    return V, truth, frame


# ------------------------------------------------------------------------------------------- ridge
def _level_shape(shape, level):
    s = tuple(int(v) for v in shape)
    for _ in range(level):
        s = tuple(v // 2 for v in s)
    return s


def _centres(idx_zyx, level):
    """Mean-pool centre mapping (d3/topo_proto.py:30): level voxel ZYX -> L0 xyz."""
    Fl = 2 ** level
    return (np.asarray(idx_zyx, np.float64)[:, ::-1] + 0.5) * Fl - 0.5


def _sheet_voxels(truth, shp, level, chunk_rows=24):
    """Voxels at ``level`` crossed by the true sheet centre (where papyrus is present): per voxel the dense
    sheet sample nearest (along the normal) to its centre, kept if that distance is <= 0.5 * 2**level *
    max|n_i| (a 26-connected digital surface).  Returns (lin, s, zm, n) sorted by linear index."""
    Fl = 2 ** level
    ds = 0.35 * Fl
    s_all = np.arange(ds / 2, truth.length, ds)
    Z = truth.shape[0]
    z_all = np.arange(-8.0, Z + 8.0, ds)
    keep = []
    shp_a = np.array(shp)
    for r0 in range(0, len(z_all), chunk_rows):
        a, b = max(r0 - 1, 0), min(r0 + chunk_rows + 1, len(z_all))
        zs = z_all[a:b]
        S, Zm = np.meshgrid(s_all, zs)
        P = truth.point(S, Zm)
        if P.shape[0] < 2:
            continue
        dPs = np.gradient(P, axis=1)
        dPz = np.gradient(P, axis=0)
        n = np.cross(dPs, dPz)
        Po = truth.point(S, Zm, 0.5)
        sg = np.sign(((Po - P) * n).sum(-1))
        n *= np.where(sg == 0, 1.0, sg)[..., None]
        with np.errstate(invalid="ignore", divide="ignore"):
            n /= np.linalg.norm(n, axis=-1, keepdims=True)
        rows = slice(r0 - a, r0 - a + min(chunk_rows, len(z_all) - r0))
        P, n, S, Zm = P[rows].reshape(-1, 3), n[rows].reshape(-1, 3), S[rows].ravel(), Zm[rows].ravel()
        ok = np.all(np.isfinite(P), 1) & np.all(np.isfinite(n), 1)
        ok[ok] = truth.present(S[ok], Zm[ok])
        P, n, S, Zm = P[ok], n[ok], S[ok], Zm[ok]
        idx = np.rint((P + 0.5) / Fl - 0.5).astype(np.int64)[:, ::-1]
        inb = np.all((idx >= 0) & (idx < shp_a), 1)
        P, n, S, Zm, idx = P[inb], n[inb], S[inb], Zm[inb], idx[inb]
        c = (idx[:, ::-1] + 0.5) * Fl - 0.5
        dist = np.abs(((c - P) * n).sum(1))
        sel = dist <= 0.5 * Fl * np.abs(n).max(1)
        if not sel.any():
            continue
        lin = np.ravel_multi_index(tuple(idx[sel].T), shp)
        o = _first_per_key(lin, dist[sel])
        keep.append((lin[o], dist[sel][o], S[sel][o], Zm[sel][o], n[sel][o]))
    if not keep:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros((0, 3))
    lin, dist, S, Zm, n = (np.concatenate(k) for k in zip(*keep))
    o = _first_per_key(lin, dist)
    return lin[o], S[o], Zm[o], n[o]


def _first_per_key(key, dist):
    """Indices of the smallest ``dist`` per distinct ``key``, in increasing key order."""
    o = np.lexsort((dist, key))
    if len(o) == 0:
        return o
    return o[np.r_[True, key[o][1:] != key[o][:-1]]]


def _ridge_contrast(vol, xyz, n, level):
    from .channels import half_prominence

    if len(xyz) == 0:
        return np.zeros(0, np.float32)
    Fl = 2 ** level
    offs = np.arange(-6.0 * Fl, 6.0 * Fl + 1e-9, 0.5 * Fl)
    prof = np.asarray(vol.sample_layers(xyz, n, offs, level=level), np.float64).T
    return half_prominence(prof, 0.5 * Fl, centre_reach=0.75 * Fl)[2].astype(np.float32)


def truth_ridge(vol, truth, frame, level: int = 0, jitter: float = 0.0, drop: float = 0.0, seed: int = 0) -> Ridge:
    """The true sheet-centre voxels at ``level`` as a ``Ridge`` (a stand-in for S1's detected ridge).

    One voxel per crossing of the true sheet centre (26-connected digital surface, papyrus present);
    ``xyz`` = the voxel's mean-pool centre, ``n`` = the true outward sheet normal, ``width`` = the true
    sheet thickness in level px, ``contrast`` = the CT half-prominence along n at that level, ``flags`` =
    PURE, plus CORE (r < frame.r_core) and RADIAL (|n . r_hat| < 0.3).  Truth never flags FUSED.
    ``drop``: remove that fraction of points at random; ``jitter``: move each point along its normal by
    N(0, jitter) L0 vx and re-quantise (duplicates merged).  Sorted by linear index (deterministic).
    The selection is in world space; ``truth.d`` is a material-space depth, so where a phantom warp shears the
    sheet a selected voxel can have |d| > 0.5 * 2**level (ws-small seed 0 at L0: 2% above 0.75 vx, max 5.8)."""
    shp = _level_shape(vol.levels[level].shape, 0)
    if shp != _level_shape(truth.shape, level):
        raise ValueError(f"volume level {level} has shape {shp}, expected {_level_shape(truth.shape, level)}")
    Fl = 2 ** level
    lin, S, Zm, n = _sheet_voxels(truth, shp, level)
    rng = np.random.default_rng(seed)
    if drop > 0 and len(lin):
        k = rng.random(len(lin)) >= drop
        lin, S, Zm, n = lin[k], S[k], Zm[k], n[k]
    if jitter > 0 and len(lin):
        xyz = _centres(np.stack(np.unravel_index(lin, shp), 1), level)
        xyz = xyz + rng.normal(0.0, jitter, len(lin))[:, None] * n
        idx = np.rint((xyz + 0.5) / Fl - 0.5).astype(np.int64)[:, ::-1]
        inb = np.all((idx >= 0) & (idx < np.array(shp)), 1)
        lin2 = np.ravel_multi_index(tuple(idx[inb].T), shp)
        o = _first_per_key(lin2, np.arange(len(lin2), dtype=np.float64))  # first occurrence wins
        sel = np.flatnonzero(inb)[o]
        lin, S, Zm, n = lin2[o], S[sel], Zm[sel], n[sel]
    idx = np.stack(np.unravel_index(lin, shp), 1).astype(np.int32)
    xyz = _centres(idx, level)
    flags = np.full(len(lin), int(F.PURE), np.uint32)
    if len(lin):
        flags[frame.radius(xyz) < frame.r_core] |= np.uint32(F.CORE)
        flags[np.abs((n * frame.rhat(xyz)).sum(1)) < 0.3] |= np.uint32(F.RADIAL)
    return Ridge(level=level, idx=idx, xyz=xyz.astype(np.float32), n=n.astype(np.float32),
                 width=(truth.thickness(S) / Fl).astype(np.float32),
                 contrast=_ridge_contrast(vol, xyz, n, level), flags=flags)


def truth_sheetlets(r: Ridge, truth, cell: int = 16) -> tuple[Sheetlets, np.ndarray]:
    """Pure sheetlets of a ridge, and the exact integer x of every node (H = vth + x).

    Ridge points are grouped per ``cell``^3 level-px cell and per 1/8-turn sector of theta by their true sheet
    (the integer round(H - H_ref - chi * wrap(theta - theta_ref) / 2 pi) relative to the group's first point),
    then split into 26-connected components.  Node id = the linear index (at the ridge's level, ZYX) of its anchor
    voxel, the smallest one (PLAN S1 item 3); ids are sorted.  vth = chi * (theta_a + wrap(theta - theta_a))
    / 2 pi in turns about the anchor's theta_a; x = round(mean(H - vth)) over the node."""
    frame = truth.frame
    if frame is None or frame.chi == 0:
        raise ValueError("truth.frame with a decided chirality is required")
    shp = _level_shape(truth.shape, r.level)
    M = len(r.idx)
    empty = Sheetlets(node=np.full(M, -1, np.int64), vth=np.full(M, np.nan), ids=np.zeros(0, np.int64),
                      size=np.zeros(0, np.int32), centroid=np.zeros((0, 3), np.float32),
                      normal=np.zeros((0, 3), np.float32), flags=np.zeros(0, np.uint32))
    if M == 0:
        return empty, np.zeros(0, np.int64)
    idx = r.idx.astype(np.int64)
    lin = np.ravel_multi_index(tuple(idx.T), shp)
    if np.any(np.diff(lin) <= 0):
        raise ValueError("ridge points must be unique and sorted by linear index (as truth_ridge returns them)")
    xyz = r.xyz.astype(np.float64)
    H = truth.H(xyz)
    th = frame.theta(xyz)
    chi = frame.chi
    cshape = tuple(-(-v // cell) for v in shp)
    # cell x angular sector: a node never spans more than 1/8 turn, so theta unwraps inside it (core wraps)
    sector = np.clip((th * (N_SECTORS / TWO_PI)).astype(np.int64), 0, N_SECTORS - 1)
    clin = np.ravel_multi_index(tuple((idx // cell).T), cshape) * N_SECTORS + sector
    o = np.lexsort((lin, clin))
    firstc = np.r_[True, clin[o][1:] != clin[o][:-1]]
    ref = np.empty(M, np.int64)
    ref[o] = o[np.maximum.accumulate(np.where(firstc, np.arange(M), 0))]
    k = np.rint(H - H[ref] - chi * wrap(th - th[ref]) / TWO_PI).astype(np.int64)
    _, grp = np.unique(np.stack([clin, k], 1), axis=0, return_inverse=True)
    grp = grp.ravel()
    # 26-connected components inside each group
    A, B = [], []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dz, dy, dx) <= (0, 0, 0):
                    continue
                q = idx + np.array([dz, dy, dx])
                inb = np.all((q >= 0) & (q < np.array(shp)), 1)
                a = np.flatnonzero(inb)
                ql = np.ravel_multi_index(tuple(q[a].T), shp)
                pos = np.clip(np.searchsorted(lin, ql), 0, M - 1)
                m = (lin[pos] == ql) & (grp[pos] == grp[a])
                A.append(a[m])
                B.append(pos[m])
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    A, B = np.concatenate(A), np.concatenate(B)
    ncomp, lab = connected_components(sp.csr_matrix((np.ones(len(A)), (A, B)), shape=(M, M)), directed=False)
    anchor_lin = np.full(ncomp, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(anchor_lin, lab, lin)
    ids = np.sort(anchor_lin)
    node = anchor_lin[lab]
    row = np.searchsorted(ids, node)
    anchor_pt = np.searchsorted(lin, ids)  # lin is sorted: the anchor's point index
    tha = th[anchor_pt][row]
    vth = chi * (tha + wrap(th - tha)) / TWO_PI
    n_nodes = len(ids)
    size = np.bincount(row, minlength=n_nodes)
    x = np.rint(np.bincount(row, H - vth, n_nodes) / size).astype(np.int64)
    cen = np.stack([np.bincount(row, xyz[:, c], n_nodes) for c in range(3)], 1) / size[:, None]
    nsum = np.stack([np.bincount(row, r.n[:, c].astype(np.float64), n_nodes) for c in range(3)], 1)
    nn = np.linalg.norm(nsum, axis=1, keepdims=True)
    nrm = np.where(nn > 0, nsum / np.where(nn > 0, nn, 1.0), 0.0)
    fl = np.zeros(n_nodes, np.uint32)
    np.bitwise_or.at(fl, row, r.flags.astype(np.uint32))
    sh = Sheetlets(node=node, vth=vth, ids=ids, size=size.astype(np.int32), centroid=cen.astype(np.float32),
                   normal=nrm.astype(np.float32), flags=fl)
    return sh, x


# ------------------------------------------------------------------------------------------- band
class TruthBand:
    """The exact band field (``BandQuery``): W = truth.H; status CONFIRMED (CORE where r < r_core, HOLE where
    W is undefined).  As d3/band_trace.py:100-106 (exact H, no error field)."""

    def __init__(self, truth, frame):
        self.truth, self.frame = truth, frame

    def W(self, p):
        p = np.asarray(p, np.float64).reshape(-1, 3)
        w = self.truth.H(p)
        st = np.full(len(p), int(Status.CONFIRMED), np.uint8)
        st[self.frame.radius(p) < self.frame.r_core] = Status.CORE
        st[~np.isfinite(w)] = Status.HOLE
        return w, st


def truth_band(truth, frame) -> BandQuery:
    return TruthBand(truth, frame)


class RefBandVolume:
    """The tracer's Volume seen through the band of one wrap: a reference port of d3/band_trace.py:86-118
    behind WP6's ``BandVolume(base, band, seed_xyz, frame)`` signature (WP7 uses it until WP6 merges).

    x is in the seed wrap's band iff |W(x) - chi * wrap(theta(x) - theta_seed) / 2 pi - W(seed)| < 0.5.
    ``sample`` and ``levels[g].read`` are unmasked (spacing / contrast calibration); ``sample_layers``, which
    every snap uses, is zero outside the band."""

    def __init__(self, base, band, seed_xyz, frame):
        self.base, self.band, self.frame = base, band, frame
        self.levels = base.levels
        s = np.asarray(seed_xyz, np.float64).reshape(1, 3)
        self.h0 = float(band.W(s)[0][0])
        self.th0 = float(frame.theta(s)[0])

    @property
    def shape(self):
        return self.base.shape

    def mask(self, p) -> np.ndarray:
        p = np.asarray(p, np.float64).reshape(-1, 3)
        h = np.asarray(self.band.W(p)[0], np.float64)
        with np.errstate(invalid="ignore"):
            return np.abs(h - self.frame.chi * wrap(self.frame.theta(p) - self.th0) / TWO_PI - self.h0) < 0.5

    def sample(self, xyz, level=0, order=1):
        return self.base.sample(xyz, level, order)

    def sample_layers(self, xyz, normals, offsets, level=0):
        xyz = np.asarray(xyz, float)
        normals = np.asarray(normals, float)
        offs = np.asarray(offsets, float).reshape(-1)
        v = self.base.sample_layers(xyz, normals, offs, level)
        pts = xyz[None] + offs.reshape((-1,) + (1,) * xyz.ndim) * normals[None]
        return v * self.mask(pts).reshape(v.shape)


def d3_seeds(vol, truth, frame, n: int = 40, rng_seed: int = 1, r_range=(50.0, 105.0)) -> np.ndarray:
    """Trace seeds on sheet centres as d3/band_trace.py:150-158 draws them: uniform in the box (20 vx from the
    x/y faces, 34 from the z faces), r in r_range, |d| <= 0.7, CT >= 60 at the voxel.  (n, 3) xyz."""
    Z, Y, X = truth.shape
    rng = np.random.default_rng(rng_seed)
    out = []
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > 200000:
            raise RuntimeError("could not draw enough seeds")
        p = np.array([rng.uniform(20, X - 20), rng.uniform(20, Y - 20), rng.uniform(34, Z - 34)])
        r_ = frame.radius(p[None])[0]
        if not (r_range[0] <= r_ <= r_range[1]) or abs(truth.d(p[None])[0]) > 0.7:
            continue
        if vol.levels[0].gather(np.array([[int(p[2]), int(p[1]), int(p[0])]]))[0] < 60:
            continue
        out.append(p)
    return np.array(out)


def score_trace(surface, seed_xyz, truth, frame) -> dict:
    """d3/band_trace.py's ``score``: per valid vertex the wrap offset k = round(H - chi * wrap(theta -
    theta_seed) / 2 pi - H_seed); on the sheet iff |d| < 3.  jump = other-wrap share > 2%."""
    P = surface.points[surface.valid].astype(np.float64)
    if len(P) == 0:
        return dict(n=0)
    s = np.asarray(seed_xyz, np.float64)[None]
    k = np.rint(truth.H(P) - frame.chi * wrap(frame.theta(P) - frame.theta(s)[0]) / TWO_PI - truth.H(s)[0])
    on = np.abs(truth.d(P)) < 3.0
    other = float(((k != 0) & on).mean())
    return dict(n=int(len(P)), same=float(((k == 0) & on).mean()), other=other, off=float((~on).mean()),
                jump=bool(other > 0.02))


# ------------------------------------------------------------------------------------------- traces
def truth_traces(truth, n: int, step: float, noise: float, jump_frac: float = 0.0, seed: int = 0) -> list[TraceRec]:
    """``n`` patches of the true sheet (the WholeScroll.true_surface construction: rows along z, columns
    along s, so du x dv points outward), (2R+1)^2 vertices with R = TRACE_RADIUS_STEPS, spacing ``step``,
    NaN where the papyrus is missing or outside the volume, plus isotropic N(0, noise) vertex noise.

    ``jump_frac``: that share of the traces (rounded) carries a planted wrap jump: from a random column on,
    the geometry lies on the adjacent wrap (t +- 2 pi) while the channel H keeps the seed wrap's value (a
    trace that silently followed the wrong sheet).  ``meta['jump']`` = {col, dw}, ``meta['jump_mask']`` the
    affected vertices.  Channels: H (claimed), width (true thickness), contrast NaN (no CT), conflict 0,
    trunc 0, plane_res (``channels.plane_res``)."""
    from .channels import plane_res

    frame = truth.frame
    rng = np.random.default_rng(seed)
    Z, Y, X = truth.shape
    R = TRACE_RADIUS_STEPS
    k_off = np.arange(-R, R + 1) * float(step)
    n_jump = int(round(jump_frac * n))
    jumpers = set(rng.permutation(n)[:n_jump].tolist()) if n_jump else set()
    t_max = float(truth.src._tc[-1])
    hi = np.array([X - 1, Y - 1, Z - 1], float)
    out = []
    for pid in range(n):
        for _ in range(10000):
            s0 = rng.uniform(0.0, truth.length)
            z0 = rng.uniform(0.0, Z - 1.0)
            p0 = truth.point(np.array([s0]), np.array([z0]))[0]
            if (np.all(np.isfinite(p0)) and np.all(p0 >= 4) and np.all(p0 <= hi - 4)
                    and truth.present(np.array([s0]), np.array([z0]))[0]
                    and frame.radius(p0[None])[0] >= frame.r_core):
                break
        else:
            raise RuntimeError("could not place a truth trace")
        ss = s0 + k_off
        zz = z0 + k_off
        rows = (zz >= 0) & (zz <= Z - 1)
        zz = zz[rows]
        S, Zm = np.meshgrid(ss, zz)
        Sg = S.copy()
        jump = None
        jmask = np.zeros(S.shape, bool)
        if pid in jumpers:
            col = int(rng.integers(R // 2, 3 * R // 2 + 1))
            t0 = float(truth.t_of_s(s0))
            dws = [k for k in (-1, 1) if 0.0 <= t0 + TWO_PI * k <= t_max] or [1]
            dw = int(dws[int(rng.integers(len(dws)))])  # a wrap that exists at the seed
            t2 = truth.t_of_s(S[:, col:]) + TWO_PI * dw
            Sg[:, col:] = np.where((t2 >= 0) & (t2 <= t_max), truth.s_of_t(np.clip(t2, 0, t_max)), np.nan)
            jmask[:, col:] = True
            jump = {"col": col, "dw": dw}
        okg = np.isfinite(Sg) & (Sg > 0) & (Sg < truth.length)
        P = np.full(S.shape + (3,), np.nan)
        P[okg] = truth.point(Sg[okg], Zm[okg])
        pres = np.zeros(S.shape, bool)
        pres[okg] = truth.present(Sg[okg], Zm[okg])
        valid = okg & pres & np.all(np.isfinite(P), -1)
        if noise > 0:
            P[valid] += rng.normal(0.0, noise, (int(valid.sum()), 3))
        valid &= np.all((P >= 0) & (P <= hi), -1)  # every valid vertex (noise included) lies inside the volume
        P[~valid] = np.nan
        surf = Surface(P.astype(np.float32), (1.0 / step, 1.0 / step),
                       {"source": "truth_traces", "step": float(step), "s0": float(s0), "z0": float(z0)})
        ch = {k: np.full(S.shape, np.nan) for k in CHANNEL_DTYPES}
        ch["H"] = np.where(valid, truth.H_of_s(S), np.nan)
        ch["width"] = np.where(valid, truth.thickness(np.where(np.isfinite(Sg), Sg, s0)), np.nan)
        ch["conflict"] = np.where(valid, 0.0, np.nan)
        ch["trunc"] = np.zeros(S.shape)
        ch["plane_res"] = plane_res(surf)
        ch = {k: v.astype(CHANNEL_DTYPES[k]) for k, v in ch.items()}
        out.append(TraceRec(surface=surf, wrap_H0=float(truth.H_of_s(s0)), seed=p0, pid=pid, channels=ch,
                            meta={"s0": float(s0), "z0": float(z0), "step": float(step), "noise": float(noise),
                                  "jump": jump, "jump_mask": jmask & valid}))
    return out


# ------------------------------------------------------------------------------------------- planted errors
def planted(sol: Solution, sheetlets, frame, kind: str, n_blocks: int, size_vx: float, seed: int) -> tuple[Solution, np.ndarray]:
    """Plant ``n_blocks`` +-1 winding errors into a copy of ``sol``.

    A block is every node of the MAIN component (outside the core) whose centroid lies within size_vx / 2
    of a random block centre (a node centroid) -- all wraps through that region shift together, like a
    coarse piece assigned one wrap off (d3/band_trace.py's topo_sharp).  Block centres are >= size_vx + 6 *
    SMOOTH_SIGMA_VX apart.  The integer x of block nodes changes by the block's sign (+1 or -1) for both
    kinds.  The second return value, per node of ``sol`` (float32), is the planted mask: for 'sharp' the
    sign inside the block and 0 outside; for 'smooth' the H ramp to apply in the band, sign * Phi((R - dist)
    / SMOOTH_SIGMA_VX) (|ramp| >= 0.5 exactly on the integer block; < 1e-3 set to 0).  ``mask != 0`` is the
    affected set; ``meta['planted']`` records centres, signs and node counts."""
    if kind not in ("sharp", "smooth"):
        raise ValueError("kind must be 'sharp' or 'smooth'")
    from scipy.special import ndtr

    rows = sheetlets.index(sol.ids)
    if np.any(rows < 0):
        raise ValueError("sol.ids are not all sheetlet ids")
    cen = sheetlets.centroid[rows].astype(np.float64)
    R = 0.5 * float(size_vx)
    cand = (sol.comp == sol.main_comp()) & (frame.radius(cen) >= frame.r_core)
    rng = np.random.default_rng(seed)
    centres, signs = [], []
    for i in rng.permutation(np.flatnonzero(cand)):
        c = cen[i]
        if all(np.linalg.norm(c - q) >= 2 * R + 6 * SMOOTH_SIGMA_VX for q in centres):
            centres.append(c)
            signs.append(int(rng.choice([-1, 1])))
            if len(centres) == n_blocks:
                break
    if len(centres) < n_blocks:
        raise ValueError(f"could place only {len(centres)} of {n_blocks} separated blocks of {size_vx} vx")
    mask = np.zeros(len(sol.ids), np.float64)
    shift = np.zeros(len(sol.ids), np.int64)
    counts = []
    for c, sg in zip(centres, signs):
        dist = np.linalg.norm(cen - c, axis=1)
        inb = cand & (dist <= R)
        shift[inb] = sg
        counts.append(int(inb.sum()))
        if kind == "sharp":
            mask[inb] = sg
        else:
            ramp = ndtr((R - dist) / SMOOTH_SIGMA_VX) * cand
            mask = np.where(np.abs(ramp) > np.abs(mask), sg * ramp, mask)
    if kind == "smooth":
        mask[np.abs(mask) < 1e-3] = 0.0
    out = sol.copy()
    out.x = (sol.x.astype(np.int64) + shift).astype(np.int32)
    out.meta["planted"] = {"kind": kind, "n_blocks": int(n_blocks), "size_vx": float(size_vx), "seed": int(seed),
                           "centres": [list(map(float, c)) for c in centres], "signs": signs, "nodes": counts,
                           "sigma_vx": SMOOTH_SIGMA_VX if kind == "smooth" else 0.0}
    return out, mask.astype(np.float32)
