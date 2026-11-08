"""Exact, geometry-only transfer of organizer ink labels from a high-resolution label scan (2.2-2.4 um) onto the
same segment's mesh in a coarse model scan (8.64 / 9.362 um).  No ink-model output and no image registration is used,
so labels made this way can be used to evaluate or train models without circularity.

Setting (Vesuvius Challenge open data, s3://vesuvius-challenge-open-data): a segment directory holds, per volume,
``<seg>-on-<VID>-<UM>um.tifxyz`` (the same parametrisation resampled to that volume's grid) and the ink label image
``ink-detection/<seg>-<UM>um-...tif`` rendered on the label-volume mesh at 20 px per grid cell.

Correspondence rule (determined from the data, verified on 5 scrolls): the '-on-' meshes are one parametrisation
resized cell-centred to each volume's grid, so a coarse-grid point g9 (row, col) maps to label-grid point
    g2 + 0.5 = (H2 / H9, W2 / W9) * (g9 + 0.5)
with the EXACT per-segment grid-size ratios (a constant voxel-size ratio such as 9.362 / 2.215 = 4.227 drifts by
several cells across a segment).  Label pixel = round(20 * g2).

Validation (in coarse-scan voxels): T is the inter-volume 3D affine (coarse xyz -> label xyz), the published
``volumes/<zarr>/transform.json`` when its fixed volume is the label volume, else a trimmed least-squares affine fitted
on the rule correspondences.  node3d = |T P9(g9) - P2(rule(g9))| at mesh nodes; a window passes if median < 1 and
p95 < 2 voxels.  When T is published, the LSQ estimate is reported as an independent cross-check.
"""
import json
import os
import re
import shutil
import time
import urllib.parse

import numpy as np
import tifffile
from scipy import ndimage as ndi

from .tifxyz import bilin, load_xyz

S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
PX = 20  # label / render pixels per grid cell (1 px = 1 voxel at mesh scale 0.05)

# Public scrolls used in the evaluation: coarse model volume, label volume id, label voxel size.
SCROLLS = {
    "PHerc0500P2": dict(v9="20250820143440-9.362um-1.2m-113keV-masked.zarr", lvid="20250526151718", um="2.215um"),
    "PHerc0814": dict(v9="20250804134230-9.362um-1.2m-113keV-masked.zarr", lvid="20260309142202", um="2.399um"),
    "PHerc0139": dict(v9="20250728140407-9.362um-1.2m-113keV-masked.zarr", lvid="20260102150214", um="2.399um"),
    "PHerc0343P": dict(v9="20250521134555-8.640um-1.2m-116keV-masked.zarr", lvid="20260304131111", um="2.215um"),
    "PHerc0009B": dict(v9="20250521125136-8.640um-1.2m-116keV-masked.zarr", lvid="20250820154339", um="2.401um"),
}


# ---------------------------------------------------------------- S3 access (public bucket, anonymous HTTP)
def s3_list(prefix):
    import requests
    out, tok = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if tok:
            q["continuation-token"] = tok
        for attempt in range(6):   # transient proxy / network errors: retry with backoff
            try:
                r = requests.get(S3 + "/", params=q, timeout=60)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                if attempt == 5:
                    raise
            time.sleep(2 ** attempt)
        t = r.text
        out += re.findall(r"<Key>([^<]*)</Key>", t)
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", t)
        if not m:
            return out
        tok = m.group(1)


def s3_get(key, dst, min_free_gb=1.5):
    import requests
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if shutil.disk_usage(os.path.dirname(dst) or ".").free < min_free_gb * 2 ** 30:
        raise RuntimeError("DISK_LOW: refusing to download " + key)
    for a in range(5):
        try:
            r = requests.get(S3 + "/" + urllib.parse.quote(key), timeout=600)
            if r.status_code == 200:
                open(dst + ".part", "wb").write(r.content)
                os.replace(dst + ".part", dst)
                return dst
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 ** a)
    return None


def fetch_segment(scroll, seg, cache, label_name=None):
    """Download the label-volume mesh, the full-resolution ink label tif, the coarse mesh and the transform for one
    segment into <cache>/<seg>/ (label mesh + label.tif), <cache>/<seg>/coarse/ (coarse mesh).  Returns paths.
    label_name: basename of the ink-detection tif to use (default: the first one in sorted order)."""
    c = SCROLLS[scroll]
    sd = os.path.join(cache, seg)
    keys = s3_list(f"{scroll}/segments/{seg}/")
    um9 = c["v9"].split("-")[1]
    lm = [k for k in keys if f"-on-{c['lvid']}-{c['um']}.tifxyz/" in k and k.endswith(("x.tif", "y.tif", "z.tif", "meta.json"))]
    vid9 = c["v9"].split("-")[0]
    # the mesh folder may spell the voxel size without trailing zeros (8.64um for the 8.640um volume)
    um_re = re.escape(vid9) + r"-" + re.escape(um9[:-2].rstrip("0").rstrip(".")) + r"0*um\.tifxyz/"
    cm = [k for k in keys if re.search(r"-on-" + um_re, k) and k.endswith(("x.tif", "y.tif", "z.tif", "meta.json"))]
    lk = [k for k in keys if "/ink-detection/" in k and "/downsampled/" not in k and f"-{c['um']}-" in k and k.endswith(".tif")]
    if label_name:
        lk = [k for k in lk if k.split("/")[-1] == label_name]
    if len([k for k in lm if k.endswith(".tif")]) != 3 or not lk:
        raise RuntimeError(f"{seg}: label mesh / label tif not found in the bucket")
    if len([k for k in cm if k.endswith(".tif")]) != 3:
        raise RuntimeError(f"{seg}: coarse {um9} mesh not found in the bucket")
    for k in lm:
        s3_get(k, os.path.join(sd, k.split("/")[-1]))
    for k in cm:
        s3_get(k, os.path.join(sd, "coarse", k.split("/")[-1]))
    lab = s3_get(sorted(lk)[0], os.path.join(sd, "label.tif"))
    tj = s3_get(f"{scroll}/volumes/{c['v9']}/transform.json", os.path.join(cache, f"transform_{um9}.json"))
    return dict(label_mesh=sd, coarse_mesh=os.path.join(sd, "coarse"), label_tif=lab, transform=tj, label_um=c["um"])


# ---------------------------------------------------------------- geometry
def rule(g9r, g9c, sh9, sh2):
    return (sh2[0] / sh9[0]) * (g9r + 0.5) - 0.5, (sh2[1] / sh9[1]) * (g9c + 0.5) - 0.5


def apply_affine(T, P):
    return P @ T[:, :3].T + T[:, 3]


def fit_affine(A, B):
    M, *_ = np.linalg.lstsq(np.c_[A, np.ones(len(A))], B, rcond=None)
    return M.T


def _stats(x):
    x = x[np.isfinite(x)]
    return dict(median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(x.max())) if x.size else None


def read_transform(path, label_um):
    """Published 3x4 affine if its fixed volume is the label volume, else None."""
    if not path or not os.path.exists(path):
        return None, ""
    d = json.load(open(path))
    fixed = d.get("fixed_volume", "")
    return (np.array(d["transformation_matrix"], np.float64)[:3], fixed) if label_um in fixed else (None, fixed)


def segment_registration(P9, P2, T_pub=None):
    """Fit / check the inter-volume affine for a segment.  Returns dict with T (used), T_est, node residual grid."""
    sh9, sh2 = P9.shape[:2], P2.shape[:2]
    v9 = np.all(np.isfinite(P9), -1); g9 = np.argwhere(v9); A = P9[v9]
    fu, fv = rule(g9[:, 0], g9[:, 1], sh9, sh2)
    X2, _, _ = bilin(P2, fu, fv)
    ok = np.all(np.isfinite(X2), 1)
    T_est = fit_affine(A[ok], X2[ok])
    for _ in range(3):  # trimmed refit
        e = np.linalg.norm(apply_affine(T_est, A) - X2, axis=1)
        keep = ok & (e <= np.nanpercentile(e[ok], 98))
        T_est = fit_affine(A[keep], X2[keep])
    T = T_pub if T_pub is not None else T_est
    s9 = abs(np.linalg.det(T[:, :3])) ** (1 / 3)  # label voxels per coarse voxel
    nres = np.full(sh9, np.nan)
    nres[v9] = np.linalg.norm(apply_affine(T, A) - X2, axis=1) / s9
    out = dict(T=T, T_est=T_est, vx_scale=float(s9), node3d=nres, T_source="published" if T_pub is not None else "lsq")
    if T_pub is not None:
        out["est_vs_published_vx"] = _stats(np.linalg.norm(apply_affine(T_est, A) - apply_affine(T_pub, A), axis=1) / s9)
    return out


def label_support(lab, sl):
    """Label support (where the organizers labelled at all): fill_holes(closing(ds8 mean > 3, 3 iterations))."""
    a = lab[sl]; H8, W8 = a.shape[0] // 8, a.shape[1] // 8
    m8 = a[:H8 * 8, :W8 * 8].reshape(H8, 8, W8, 8).mean((1, 3))
    return ndi.binary_fill_holes(ndi.binary_closing(m8 > 3, iterations=3))


def transfer_crop(P9, P2, lab, reg, r0, c0, rows, cols, px=PX):
    """Labels for the coarse-mesh crop spanning `rows` x `cols` grid cells starting at (r0, c0), rendered at `px` pixels per
    cell (20 = 1 px per voxel at scale 0.05).  Pixel (py, px_) is grid point (r0 + py/px, c0 + px_/px).
    The crop includes both end nodes: rows * px + 1 by cols * px + 1 pixels.
    Returns ink_raw (uint8, nearest label px), acc (label support & label mesh valid), valid (coarse mesh valid),
    validation dict."""
    sh9, sh2 = P9.shape[:2], P2.shape[:2]
    H, W = rows * px + 1, cols * px + 1
    pr, pc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    gr = (r0 + pr / px).ravel(); gc = (c0 + pc / px).ravel()
    X9, _, _ = bilin(P9, gr, gc)
    valid = np.all(np.isfinite(X9), 1)
    uu, vv = rule(gr, gc, sh9, sh2)
    Y2, _, _ = bilin(P2, uu, vv)
    pix3d = np.linalg.norm(Y2 - apply_affine(reg["T"], X9), axis=1) / reg["vx_scale"]
    pix3d[~valid] = np.nan
    yy = np.rint(uu * px).astype(np.int64); xx = np.rint(vv * px).astype(np.int64)
    inl = (yy >= 0) & (yy < lab.shape[0]) & (xx >= 0) & (xx < lab.shape[1])
    yy = np.where(inl, yy, 0); xx = np.where(inl, xx, 0)
    y0 = max(int(yy[inl].min()) - 64, 0); y1 = min(int(yy[inl].max()) + 65, lab.shape[0])
    x0 = max(int(xx[inl].min()) - 64, 0); x1 = min(int(xx[inl].max()) + 65, lab.shape[1])
    y0 -= y0 % 8; x0 -= x0 % 8
    sup = label_support(lab, (slice(y0, y1), slice(x0, x1)))
    sy = np.clip((yy - y0) // 8, 0, sup.shape[0] - 1); sx = np.clip((xx - x0) // 8, 0, sup.shape[1] - 1)
    raw = np.where(inl, lab[yy, xx], 0).astype(np.uint8).reshape(H, W)
    acc = (inl & sup[sy, sx] & np.all(np.isfinite(Y2), 1)).reshape(H, W)
    wn = reg["node3d"][r0:r0 + rows + 1, c0:c0 + cols + 1]
    val = dict(node3d_vx=_stats(wn.ravel()), pix3d_vx=_stats(pix3d))
    val["pass"] = bool(val["node3d_vx"] and val["node3d_vx"]["median"] < 1 and val["node3d_vx"]["p95"] < 2)
    return raw, acc, valid.reshape(H, W), val


def transfer(coarse_mesh, label_mesh, label_tif, transform=None, label_um="", crop=None, px=PX):
    """Full pipeline for one segment.  crop = (r0, c0, rows, cols) in coarse grid cells, default the whole grid."""
    P9 = load_xyz(coarse_mesh); P2 = load_xyz(label_mesh)
    lab = tifffile.imread(label_tif)
    if lab.ndim == 3:
        lab = lab[..., 0]
    sh2 = P2.shape[:2]
    if lab.shape != (sh2[0] * px, sh2[1] * px):
        raise ValueError(f"label image {lab.shape} is not {px} x label grid {sh2}")
    T_pub, fixed = read_transform(transform, label_um) if transform else (None, "")
    reg = segment_registration(P9, P2, T_pub)
    if crop is None:
        crop = (0, 0, P9.shape[0] - 1, P9.shape[1] - 1)
    raw, acc, valid, val = transfer_crop(P9, P2, lab, reg, *crop, px=px)
    seg_meta = dict(T=reg["T"].tolist(), T_source=reg["T_source"] + ((" (fixed: %s)" % fixed) if T_pub is not None else ""),
                    T_est=reg["T_est"].tolist(), est_vs_published_vx=reg.get("est_vs_published_vx"),
                    grid_coarse=list(P9.shape[:2]), grid_label=list(sh2), crop=list(crop), px_per_cell=px,
                    rule="g2 = (%d/%d, %d/%d) * (g9 + 0.5) - 0.5; label px = round(%d * g2)" % (sh2[0], P9.shape[0], sh2[1], P9.shape[1], px),
                    segment_node3d_vx=_stats(reg["node3d"].ravel()), validation=val)
    return raw, acc, valid, seg_meta
