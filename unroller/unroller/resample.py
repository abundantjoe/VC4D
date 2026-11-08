"""Turn a parameterisation (UV per vertex) back into a regular tifxyz grid.

Every output pixel centre is located in the UV triangulation and its 3D position is the
barycentric interpolation of the triangle corners (like VC3D's abfFlattenToNewSurface).
Also used to import OBJ meshes that carry texture coordinates.
"""
from __future__ import annotations

import numpy as np

from .surface import Surface


def rasterize(V: np.ndarray, T: np.ndarray, uv: np.ndarray, step: float, origin=None, shape=None):
    """Barycentric rasterisation of mesh (V, T) laid out at ``uv``.

    Output pixel (r, c) samples uv = origin + (c*step, r*step).  Returns (H, W, 3) with NaN outside.
    """
    uvT = uv[T]  # (nT,3,2)
    ok = np.all(np.isfinite(uvT.reshape(len(T), -1)), axis=1)
    T, uvT = T[ok], uvT[ok]
    if origin is None or shape is None:
        used = np.zeros(len(uv), bool)  # vertices referenced by a kept triangle (min/max: order-free,
        used[T.ravel()] = True          # so a mask gives what uv[np.unique(T)] gave, without sorting)
        uvu = uv[used]
    if origin is None:
        origin = np.nanmin(uvu, axis=0)
    origin = np.asarray(origin, float)
    if shape is None:
        ext = np.nanmax(uvu, axis=0) - origin
        shape = (int(np.floor(ext[1] / step)) + 1, int(np.floor(ext[0] / step)) + 1)
    H, W = shape
    from . import _kernels as K

    out = np.full((H, W, 3), np.nan, np.float32)
    best = np.full((H, W), -np.inf, np.float32)  # keep the most "inside" triangle at shared edges
    g = np.ascontiguousarray((uvT - origin) / step, dtype=np.float64)  # pixel coords (x=col, y=row)
    K.rasterize_tris_banded(g, np.ascontiguousarray(T), np.ascontiguousarray(V, dtype=np.float64), out, best,
                            max(8, -(-H // 64)))
    return out


def flat_surface(surface: Surface, uv_grid: np.ndarray, out_scale: float | None = None) -> Surface:
    """Resample ``surface`` onto a regular grid in its flattened (UV, voxel units) domain."""
    idx, V, T = surface.triangles()
    uv = uv_grid[idx >= 0]
    s = out_scale if out_scale is not None else float(np.sqrt(surface.scale[0] * surface.scale[1]))
    filled = surface.filled_mask()[idx >= 0]
    if filled.any():  # rasterise a filled-vertex indicator on the same output grid
        ok = np.all(np.isfinite(uv[T].reshape(len(T), -1)), axis=1)
        used = np.zeros(len(uv), bool)
        used[T[ok].ravel()] = True
        origin = np.nanmin(uv[used], axis=0)
        ext = np.nanmax(uv[used], axis=0) - origin
        shape = (int(np.floor(ext[1] / (1.0 / s))) + 1, int(np.floor(ext[0] / (1.0 / s))) + 1)
        pts = rasterize(V, T, uv, step=1.0 / s, origin=origin, shape=shape)
        ind = rasterize(np.c_[filled, filled, filled].astype(float), T, uv, step=1.0 / s, origin=origin, shape=shape)
    else:
        pts = rasterize(V, T, uv, step=1.0 / s)
    meta = {k: v for k, v in surface.meta.items() if k not in ("uuid", "bbox", "filled", "folds_removed")}
    meta["flattened_from"] = surface.meta.get("uuid")
    if filled.any():  # a pixel interpolated from any filled vertex is not traced evidence either
        meta["filled"] = {"grid": list(pts.shape[:2]),
                          "flat_index": np.flatnonzero(np.nan_to_num(ind[..., 0]) > 1e-6).tolist()}
    return Surface(pts, (s, s), meta)


def read_obj(path: str):
    """Minimal OBJ reader: returns V (N,3), VT (M,2) or None, F (T,3) vertex idx, FT (T,3) uv idx or None.

    Polygons are fan-triangulated.
    """
    V, VT, F, FT = [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                VT.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f "):
                toks = [t.split("/") for t in line.split()[1:]]
                vi = [int(t[0]) for t in toks]
                ti = [int(t[1]) if len(t) > 1 and t[1] else 0 for t in toks]
                for k in range(1, len(vi) - 1):
                    F.append([vi[0], vi[k], vi[k + 1]])
                    FT.append([ti[0], ti[k], ti[k + 1]])
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    F = np.where(F < 0, F + len(V) + 1, F) - 1
    FT = np.asarray(FT, np.int64)
    if VT and FT.min() > 0:
        VT = np.asarray(VT, float)
        FT = FT - 1
    else:
        VT, FT = None, None
    return V, VT, F, FT


def obj_to_surface(path: str, step: float = 1.0, uv_size=None) -> Surface:
    """Convert an OBJ with texture coords into tifxyz by rasterising in UV space.

    ``uv_size`` = (width, height) in pixels that UV [0,1] spans (e.g. the segment's .tif size); if
    omitted, per-axis UV scales are fitted to the 3D edge lengths (so one output pixel ~ ``step`` voxels).
    """
    V, VT, F, FT = read_obj(path)
    if VT is None:
        raise ValueError("OBJ has no texture coordinates; flatten it first")
    # split vertices on UV seams: one 3D position per (v, vt) pair
    pairs = np.stack([F.ravel(), FT.ravel()], 1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    V2, uv = V[uniq[:, 0]], VT[uniq[:, 1]].copy()
    uv[:, 1] = 1.0 - uv[:, 1]  # OBJ v points up, image rows point down
    T = inv.reshape(-1, 3)
    if uv_size is not None:
        uv *= np.asarray(uv_size, float)
    else:
        # UVs are often normalised per axis to [0,1]; recover per-axis scales (su, sv) from the
        # 3D edge lengths: |L|^2 = su^2 du^2 + sv^2 dv^2, linear least squares in (su^2, sv^2).
        E = np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]
        d = uv[E[:, 1]] - uv[E[:, 0]]
        L2 = ((V2[E[:, 1]] - V2[E[:, 0]]) ** 2).sum(1)
        coef, *_ = np.linalg.lstsq(d**2, L2, rcond=None)
        uv *= np.sqrt(np.clip(coef, 1e-12, None))
    pts = rasterize(V2, T, uv, step=step)
    return Surface(pts, (1.0 / step, 1.0 / step), {"source": path})
