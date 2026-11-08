"""Minimal tifxyz helpers (pure numpy): load a grid, VC3D-convention normals, bilinear sampling, closest-point projection.

tifxyz = a directory with x.tif, y.tif, z.tif (float32, H x W), one 3D point (voxel coordinates of the volume the mesh
was traced in) per grid node; invalid nodes are -1 (or <= 0 / non-finite).  meta.json usually holds ``scale``
(grid cells per voxel, e.g. 0.05 = one node every 20 voxels).
"""
import json
import os

import numpy as np
import tifffile


def load_xyz(d):
    """(H, W, 3) float64 grid; invalid nodes -> NaN."""
    P = np.stack([tifffile.imread(os.path.join(d, c + ".tif")).astype(np.float64) for c in "xyz"], -1)
    P[np.all(P <= 0, -1) | ~np.all(np.isfinite(P), -1)] = np.nan
    return P


def load_meta(d):
    f = os.path.join(d, "meta.json")
    return json.load(open(f)) if os.path.exists(f) else {}


def vc_normals(P):
    """Unit normals in the VC3D / volume-cartographer convention: n = normalize(dP/dcol x dP/drow)
    (Geometry.cpp grid_normal: xv = right - left along columns, yv = down - up along rows, n = xv x yv).
    Central differences where both neighbours are valid, one-sided otherwise, NaN where a direction has no neighbour.
    vc_render_tifxyz renders slice k at offset (k - (n-1)/2) * step along this n, i.e. layer index increases along +n."""
    def deriv(axis):
        a = np.moveaxis(P, axis, 0)
        nanpad = np.full_like(a[:1], np.nan)
        nxt = np.concatenate([a[1:], nanpad])                 # a[i+1]
        prv = np.concatenate([nanpad, a[:-1]])                # a[i-1]
        cen, fwd, bwd = (nxt - prv) / 2.0, nxt - a, a - prv
        fin = lambda x: np.all(np.isfinite(x), -1, keepdims=True)
        d = np.where(fin(cen), cen, np.where(fin(fwd), fwd, bwd))
        return np.moveaxis(d, 0, axis)
    dcol, drow = deriv(1), deriv(0)
    n = np.cross(dcol, drow)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / ln
    n[~(ln[..., 0] > 0) | ~np.all(np.isfinite(P), -1)] = np.nan
    return n


def bilin(P, u, v):
    """Bilinear point and partials of grid P (H, W, 3) at fractional (row u, col v); NaN unless all 4 corners valid."""
    H, W = P.shape[:2]
    i = np.clip(np.floor(u).astype(np.int64), 0, H - 2)
    j = np.clip(np.floor(v).astype(np.int64), 0, W - 2)
    a = (u - i)[:, None]
    b = (v - j)[:, None]
    p00, p01, p10, p11 = P[i, j], P[i, j + 1], P[i + 1, j], P[i + 1, j + 1]
    X = p00 * (1 - a) * (1 - b) + p01 * (1 - a) * b + p10 * a * (1 - b) + p11 * a * b
    Du = (p10 - p00) * (1 - b) + (p11 - p01) * b
    Dv = (p01 - p00) * (1 - a) + (p11 - p10) * a
    out = (u < 0) | (u > H - 1) | (v < 0) | (v > W - 1)
    X[out] = np.nan
    return X, Du, Dv


def project(P2, tree, nodes2, Q, it=12):
    """Closest point on the bilinear surface P2 to each Q (n, 3) by Gauss-Newton from the nearest node.
    Returns (u, v, dist)."""
    _, k = tree.query(Q, workers=2)
    u = nodes2[k, 0].astype(np.float64)
    v = nodes2[k, 1].astype(np.float64)
    H, W = P2.shape[:2]
    for _ in range(it):
        X, Du, Dv = bilin(P2, u, v)
        r = X - Q
        a11 = (Du * Du).sum(1); a12 = (Du * Dv).sum(1); a22 = (Dv * Dv).sum(1)
        g1 = (Du * r).sum(1); g2 = (Dv * r).sum(1)
        det = a11 * a22 - a12 ** 2
        ok = np.isfinite(det) & (det > 1e-9)
        sd = np.where(ok, det, 1)
        du = np.where(ok, -(a22 * g1 - a12 * g2) / sd, 0)
        dv = np.where(ok, -(a11 * g2 - a12 * g1) / sd, 0)
        u = np.clip(u + np.clip(du, -1, 1), 0, H - 1)
        v = np.clip(v + np.clip(dv, -1, 1), 0, W - 1)
    X, _, _ = bilin(P2, u, v)
    return u, v, np.linalg.norm(X - Q, axis=1)
