"""Grow a new segment from a seed point (for scrolls nobody has segmented yet).

Generation by generation, every frontier cell is predicted from a weighted least-squares plane through
the placed vertices around it (or extrapolated from its valid neighbours, p = 2*n1 - n2 in the tangent
plane, where the window is too sparse), snapped along the plane normal to the sheet's density mode within
+-snap_radius (mean-shift confined between the dark gaps on either side), checked against its
neighbourhood (distance to the plane, bright-run width), then the newest rings are relaxed (edge springs
at length ``step`` + straightness) and snapped again.  snap_radius must stay below half the local sheet
spacing, otherwise the trace can jump to the neighbouring wrap: this is the dominant failure mode of all
tracers, and ``trace_report`` / the phantom benchmark measure it explicitly.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation

from .surface import Surface
from .volume import Volume


def seed_normal(vol: Volume, p, r: int = 6, level: int = 0) -> np.ndarray:
    """Sheet normal at p from the gradient structure tensor (dominant gradient direction)."""
    x, y, z = np.round(p).astype(int)
    box = vol.levels[level].read(z - r, z + r + 1, y - r, y + r + 1, x - r, x + r + 1).astype(float)
    box = gaussian_filter(box, 1.0)
    gz, gy, gx = np.gradient(box)
    g = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    w, v = np.linalg.eigh(g.T @ g)
    return v[:, -1]


def _peaks(prof, offs, rel_prominence=0.2):
    """Local maxima of a 1D profile with prominence >= rel_prominence * strongest prominence."""
    from scipy.signal import find_peaks

    pk, pr = find_peaks(prof, prominence=0)
    if len(pk) == 0:
        return np.array([])
    keep = pr["prominences"] >= rel_prominence * pr["prominences"].max()
    return offs[pk[keep]]


def estimate_spacing(vol: Volume, p, n, reach: float = 32.0, level: int = 0) -> float:
    """Distance to the neighbouring sheet along n at p (voxels), conservative by construction.

    Two estimators on the density profile, and the SMALLER is returned:
      * nearest prominent peak >= 5 vx from the seed's own peak (5 vx: closer peaks are the other
        fiber layer of the same sheet);
      * first strong autocorrelation maximum (the sheet period).
    Calibrated on 229 real Scroll 1 probes against human sheet labels: autocorrelation alone
    overestimated by >1.5x in 22% of probes (it can lock onto a harmonic), nearest-peak in 5%.
    Overestimating makes the snap radius large enough to reach the next wrap, so we take the min:
    a wrap jump then needs both to fail; the residual error (underestimation) only stalls growth.
    NaN if neither finds a neighbouring sheet within +-reach.
    """
    from scipy.signal import find_peaks

    offs = np.arange(-reach, reach + 1e-9, 0.5)
    raw = np.nan_to_num(vol.sample(np.asarray(p) + offs[:, None] * np.asarray(n), level=level))
    est = []
    sm = gaussian_filter(raw, 3.0)  # 1.5 vx
    pk, pr = find_peaks(sm, prominence=0)
    if len(pk):
        po = offs[pk[pr["prominences"] >= 0.15 * pr["prominences"].max()]]
        c = po[np.argmin(np.abs(po))]
        rest = np.abs(po[np.abs(po - c) >= 5.0] - c)
        if len(rest):
            est.append(float(rest.min()))
    f = gaussian_filter(raw, 1.0)
    f = f - f.mean()
    if np.any(f):
        ac = np.correlate(f, f, mode="full")[len(f) - 1:]
        ac = ac / ac[0]
        lags = np.arange(len(ac)) * 0.5
        apk, _ = find_peaks(ac, height=0.15, prominence=0.1)
        apk = apk[lags[apk] >= 5.0]
        if len(apk):
            est.append(float(lags[apk[0]]))
    return float(min(est)) if est else float("nan")


def _rotate_like(v, a, b):
    """Apply to vectors v the rotations taking unit vectors a -> b (row-wise, Rodrigues)."""
    k = np.cross(a, b)
    s = np.linalg.norm(k, axis=1, keepdims=True)
    c = (a * b).sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(s > 1e-9, k / s, 0.0)
    return v * c + np.cross(k, v) * s + k * (k * v).sum(1, keepdims=True) * (1 - c)


def _bright_width(prof, im, step):
    """Width (voxels) of the contiguous above-half-maximum run containing index im, per row."""
    K = prof.shape[1]
    base = prof.min(1, keepdims=True)
    peak = np.take_along_axis(prof, im[:, None], 1)
    above = prof > base + 0.5 * (peak - base)
    idx = np.arange(K)[None, :]
    left = np.maximum.accumulate(np.where(~above, idx, -1), axis=1)
    right = np.minimum.accumulate(np.where(~above, idx, K)[:, ::-1], axis=1)[:, ::-1]
    li = np.take_along_axis(left, im[:, None], 1)[:, 0]
    ri = np.take_along_axis(right, im[:, None], 1)[:, 0]
    return (ri - li - 1) * step


def _snap_ms(vol, p, n, radius, sigma, min_contrast=10.0, level=0, iters=8, step=0.5, clip=None,
             max_width=None, return_width=False, tol=0.05, basin=None, basin_smooth=1.0, strict=True):
    """Sheet-level snap: mean-shift of the density along n with a Gaussian kernel of width sigma.

    With sigma ~ spacing/4 both fiber layers of one sheet fall in the kernel, so the mode is the
    sheet's middle -- a vertex cannot hop between recto and verso.  Displacement is clipped to
    +-radius (< half the sheet spacing), so it cannot be pulled to the next wrap.
    The normal profile is sampled ONCE over +-(radius + 3 sigma); mean-shift then runs in memory,
    at most ``iters`` iterations, until no vertex moves by ``tol`` or more.
    ``clip`` caps densities above the papyrus range so a dense foreign body (an insect, a hair, a
    mineral grain lodged between wraps) cannot drag the mode off the sheet.
    ``basin`` (a density prominence): the weights and the displacement are confined to the basin of the
    start point, between the nearest dark minima on either side that are >= basin deep
    (snap_fast.basin_bounds): the mode cannot creep across a gap into the next wrap.
    ``strict``: the mode must lie strictly inside +-radius (False: the contrast and width tests alone).
    This is the numpy reference; the tracer runs the bit-identical compiled snap_fast.snap_ms.
    """
    from .snap_fast import basin_bounds

    reach = radius + 3 * sigma
    offs = np.arange(-reach, reach + 1e-9, step)
    pts = p[:, None, :] + offs[None, :, None] * n[:, None, :]  # one vectorised sample call
    prof = np.nan_to_num(vol.sample(pts.reshape(-1, 3), level=level).reshape(len(p), len(offs)))
    if clip is not None:  # dense intruders (insects, minerals) must not outvote the papyrus
        prof = np.minimum(prof, clip)
    prof = prof - prof.min(1, keepdims=True)
    wprof, lo, hi = prof, -radius, radius
    if basin is not None:
        bl, bh = basin_bounds(prof, step, basin, basin_smooth)
        idx = np.arange(len(offs))[None, :]
        wprof = np.where((idx >= bl[:, None]) & (idx <= bh[:, None]), prof, 0.0)
        lo, hi = np.maximum(-radius, offs[bl]), np.minimum(radius, offs[bh])
    d = np.zeros(len(p))
    for _ in range(iters):
        w = np.exp(-0.5 * ((offs[None, :] - d[:, None]) / sigma) ** 2) * wprof
        sw = w.sum(1)
        dn = np.where(sw > 0, (w * offs[None, :]).sum(1) / np.maximum(sw, 1e-9), d)
        dn = np.clip(dn, lo, hi)
        done = len(p) == 0 or np.max(np.abs(dn - d)) < tol
        d = dn
        if done:
            break
    # contrast: kernel-weighted density at the mode vs the darkest point of the profile
    g = np.exp(-0.5 * ((offs[None, :] - d[:, None]) / sigma) ** 2)
    at_mode = (prof * g).sum(1) / g.sum(1)
    ok = at_mode >= min_contrast
    if strict:
        ok &= np.abs(d) < radius - 1e-6
    width = None
    if max_width is not None or return_width:
        # a papyrus sheet has a bounded thickness; an insect body, a hair bundle or a fused stack of
        # several sheets is much wider -> refuse to snap there (leave a hole) rather than ride over it
        im = np.clip(np.rint((d + reach) / step).astype(int), 0, len(offs) - 1)
        width = _bright_width(prof, im, step)
        if max_width is not None:
            ok &= width <= max_width
    if return_width:
        return p + d[:, None] * n, ok, width
    return p + d[:, None] * n, ok


def _snap(vol, p, n, radius, step=0.5, min_contrast=10.0, level=0, margin=4.0):
    """Move points p along n to the density peak within +-radius.  Returns (new p, ok).

    The profile is sampled over +-(radius + margin) so that contrast is measured against the
    surrounding background, but only a true local maximum inside +-radius is accepted.
    """
    offs = np.arange(-(radius + margin), radius + margin + 1e-9, step)
    prof = np.stack([vol.sample(p + o * n, level=level) for o in offs])  # (K, N)
    prof = gaussian_filter(np.nan_to_num(prof), (1.0, 0))  # smooth along the profile only
    K, N_ = prof.shape
    inner = np.abs(offs) <= radius + 1e-9
    i = np.argmax(np.where(inner[:, None], prof, -np.inf), 0)
    cols = np.arange(N_)
    a, b, c = prof[i - 1, cols], prof[i, cols], prof[i + 1, cols]
    is_max = (b >= a) & (b >= c)
    contrast = b - prof.min(0)
    den = a - 2 * b + c
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(den < 0, 0.5 * (a - c) / den, 0.0)
    d = offs[i] + np.clip(frac, -0.5, 0.5) * step
    ok = is_max & (contrast >= min_contrast) & np.isfinite(d)
    return p + np.where(ok, d, 0)[:, None] * n, ok


# trace(..., **LEGACY) is the tracer without the wrap-jump safeguards, bit for bit
LEGACY = dict(plane_fit=0, probes=0.0, width_rel=None, plane_gate=None, basin=None, ms_iters=8, ms_tol=0.05,
              relax_soft=False, seed_check=False)


class SeedRejected(ValueError):
    """The seed does not look like the sheets around it (see ``trace``'s ``seed_check``)."""


def _probes(vol, seed, n0, u_ax, v_ax, reach):
    """The seed and up to 24 probes on its tangent plane (rings at reach/3, 2 reach/3, reach; 8 directions),
    each with its own structure-tensor normal (sign of n0)."""
    pts, nrms = [seed], [n0]
    hi = np.array(vol.shape[::-1]) - 9
    for rad in (reach / 3, 2 * reach / 3, reach):
        for k in range(8):
            a = 2 * np.pi * k / 8
            q = seed + rad * (np.cos(a) * u_ax + np.sin(a) * v_ax)
            if np.any(q < 8) or np.any(q > hi):
                continue
            nq = seed_normal(vol, q)
            pts.append(q)
            nrms.append(-nq if nq @ n0 < 0 else nq)
    return np.array(pts), np.array(nrms)


SEED_OUTLIER = 2.5  # x the probes' median width or spacing (the factor of the global width gate)


def _seed_check(w0, ws, s0, ss):
    """Is the seed unlike the sheet around it?  w0, s0: its bright-run width and spacing; ws, ss: the probes'.
    outlier: a width > SEED_OUTLIER x the probes' median (its own snap would fail the width gate) or a spacing
    off their median by more than that factor -- a seed on a foreign body (phantom anomaly case 2), but also
    3 of 216 real Scroll 1 seeds (2 of them on good sheets) and 1 of 21 held-out sector phantoms without
    foreign bodies.  suspect: no spacing at the seed where most probes have one -- a seed against a foreign
    body (anomaly case 5), but also 12 of 216 real seeds.  The pre-registered rule counted "suspect" as an outlier
    (15 of 216 real seeds refused, 8 of them tracing well); it was demoted after those counts were seen (post hoc)."""
    wm = float(np.median(ws[np.isfinite(ws)])) if np.isfinite(ws).any() else np.nan
    fin = np.isfinite(ss)
    sm = float(np.median(ss[fin])) if fin.mean() >= 0.5 else np.nan
    why = []
    if w0 > SEED_OUTLIER * wm:
        why.append(f"bright-run width {w0:.1f} vx > {SEED_OUTLIER} x the probes' median {wm:.1f} vx")
    if np.isfinite(sm) and np.isfinite(s0) and not sm / SEED_OUTLIER <= s0 <= SEED_OUTLIER * sm:
        why.append(f"sheet spacing {s0:.1f} vx vs the probes' median {sm:.1f} vx")
    suspect = bool(np.isfinite(sm) and not np.isfinite(s0))
    num = lambda v: round(float(v), 3) if np.isfinite(v) else None
    return {"seed_width": num(w0), "probe_width": num(wm), "seed_spacing": num(s0), "probe_spacing": num(sm),
            "probes": int(len(ws)), "outlier": bool(why), "reason": "; ".join(why), "suspect": suspect,
            "note": f"no neighbouring sheet found at the seed (the probes' median spacing is {sm:.1f} vx)"
            if suspect else ""}


def _plane_fit(P, fr, fc, R):
    """Weighted least-squares plane P(dc, dr) = a + b dc + c dr through the valid vertices of the (2R+1)^2
    window around each cell (fr, fc), Gaussian weights sigma = R/1.5 cells.  Returns (a = the plane at the
    cell, unit normal b x c, good): good needs >= 4 vertices spanning both grid directions."""
    from . import _kernels as K

    sig = R / 1.5
    k = np.arange(-R, R + 1)
    wt = np.exp(-0.5 * (k[:, None] ** 2 + k[None, :] ** 2) / sig ** 2)
    n = len(fr)
    M, B, cnt = np.empty((n, 3, 3)), np.empty((n, 3, 3)), np.empty(n, np.int64)
    K.call(K.plane_moments, n * wt.size, P, fr, fc, wt, M, B, cnt)
    good = (cnt >= 4) & (np.abs(np.linalg.det(M)) > 1e-6 * np.maximum(M[:, 0, 0], 1e-9) ** 3)
    a, nn = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    if good.any():
        X = np.linalg.solve(M[good], B[good])  # (k, 3 parameters, xyz)
        a[good] = X[:, 0]
        cr = np.cross(X[:, 1], X[:, 2])
        ln = np.linalg.norm(cr, axis=1)
        flat = ln > 1e-9
        tmp = np.full((int(good.sum()), 3), np.nan)
        tmp[flat] = cr[flat] / ln[flat, None]
        nn[good] = tmp
        good[np.nonzero(good)[0][~flat]] = False
    return a, nn, good


def trace(vol: Volume, seed, step: float = 4.0, generations: int = 60, snap_radius: float | None = None,
          sheet_sigma: float | None = None, snap: str = "meanshift", clip_density: float | None = None,
          max_width_factor: float | None = 2.5, curvature: bool = False,
          max_size: int = 512, min_contrast: float | None = None, relax: int = 3, bounds=None,
          up=(0.0, 0.0, 1.0), umbilicus=None, log=None, plane_fit: int = 3, probes: float = 24.0,
          width_rel: float | None = 1.5, plane_gate: float | None = 0.25, basin: float | None = 0.15,
          ms_iters: int = 8, ms_tol: float = 0.05, relax_soft: bool = False,
          seed_check: bool = False) -> Surface:
    """Grow a tifxyz grid of spacing ``step`` voxels from ``seed`` (x, y, z, level-0 voxels).

    ``up`` is the direction the grid's v (row) axis is aligned with -- the scroll axis.
    ``umbilicus`` is any point (x, y, z) on the scroll axis (default: centre of the volume). It fixes
    the sign of the normal to point *away from the axis*, which fixes which face is rendered at +offsets
    and the handedness of the grid -- without it, text could come out mirrored or from the wrong face.
    ``bounds`` = ((x0,y0,z0),(x1,y1,z1)) stops growth outside a box.
    ``snap_radius`` None = calibrate from the local sheet period (0.35 x period, in [1, 5] vx):
    a search narrower than half the period cannot reach the neighbouring wrap.
    ``snap`` = "meanshift" (default; sheet-level mode with kernel ``sheet_sigma``, default period/4,
    so recto and verso of one sheet form a single mode) or "peak" (nearest density maximum).
    ``max_width_factor``: snaps onto a bright region wider than this factor x the sheet width
    measured at the seed are refused (foreign bodies such as insects, fused bundles): the trace
    leaves a hole there instead of following the intruder.

    Wrap-jump safeguards (``**LEGACY`` switches them all off):
    ``plane_fit`` R: a new vertex is predicted, and snapped along the normal of, a weighted least-squares
    plane through the placed vertices of its (2R+1)^2 grid window (_plane_fit; fallback: the two-point
    extrapolation).  One-sided differences at the front tilt the normal by 20-85 deg where jumps start.
    ``probes`` (vx): calibrate from the seed and 24 probes on its tangent plane out to this distance:
    spacing = their 25th percentile, sheet width = their median, so one foreign body or one wide gap at
    the seed cannot inflate the snap radius or the width gate.  0 = the seed alone.
    ``width_rel``: refuse a new vertex whose bright run is wider than width_rel x the median width of the
    vertices placed within 2 cells (floor 2 vx) -- a hole instead of riding onto a fused contact.  It also stalls
    some traces on good sheets: on real Scroll 1, 19 of 145 cleanly traced seeds keep < half the previous tracer's
    own-sheet coverage (12 of 145 at 60 generations; 1 of 145 without this gate).
    ``plane_gate``: refuse a new vertex farther than plane_gate x spacing from its fitted plane.
    ``basin``: confine the mean-shift between the dark minima >= basin x the seed's contrast deep on either
    side of the prediction (_snap_ms): it cannot ride one fiber layer of a delaminated sheet into the gap.
    ``ms_iters`` / ``ms_tol``: mean-shift iterations / stop once no vertex moves by ms_tol.
    ``relax_soft``: the relax re-snap also keeps a vertex whose mode reaches the edge of its +-1.5 vx search.
    meta["seed_check"] flags a seed whose bright-run width or spacing is an outlier against the probes'
    medians (a seed on a foreign body or a fused contact) or that has no measurable spacing ("suspect"),
    and ``log`` warns; ``seed_check`` = True raises SeedRejected on an outlier instead (off by default: it also
    fires on seeds that trace well, see _seed_check).
    """
    from .snap_fast import snap_ms

    if seed_check and not probes:
        raise ValueError("seed_check compares the seed with its probes: it needs probes > 0")
    seed = np.asarray(seed, float)
    n0 = seed_normal(vol, seed)
    up = np.asarray(up, float) / np.linalg.norm(up)
    if umbilicus is None:
        sh = vol.shape
        umbilicus = ((sh[2] - 1) / 2, (sh[1] - 1) / 2, (sh[0] - 1) / 2)
    radial = seed - np.asarray(umbilicus, float)
    radial -= (radial @ up) * up
    if n0 @ radial < 0:
        n0 = -n0
    v_ax = up - (up @ n0) * n0
    if np.linalg.norm(v_ax) < 1e-3:
        v_ax = np.cross(n0, [1.0, 0, 0])
    v_ax /= np.linalg.norm(v_ax)
    u_ax = np.cross(v_ax, n0)  # so that du x dv = n0 (VC3D normal convention)
    spacing = spacing0 = estimate_spacing(vol, seed, n0)
    if probes:
        qp, qn = _probes(vol, seed, n0, u_ax, v_ax, probes)
        sps = np.array([spacing0] + [estimate_spacing(vol, q, nq) for q, nq in zip(qp[1:], qn[1:])])
        if np.isfinite(sps).any():
            spacing = float(np.percentile(sps[np.isfinite(sps)], 25))
    if snap_radius is None:
        snap_radius = float(np.clip(0.35 * spacing, 1.0, 5.0)) if np.isfinite(spacing) else 2.0
    if sheet_sigma is None:
        sheet_sigma = float(np.clip(spacing / 4, 1.0, 4.0)) if np.isfinite(spacing) else 1.5
    if clip_density is None:  # papyrus bulk density: 95th pct of non-air voxels in a block around the seed
        xs, ys, zs = np.round(seed).astype(int)
        blk = vol.levels[0].read(zs - 24, zs + 24, ys - 24, ys + 24, xs - 24, xs + 24).astype(float)
        pos = blk[blk > np.percentile(blk, 50)]
        clip_density = float(np.percentile(pos, 95)) if pos.size else None

    def width_at(q, nq):
        _, okq, wq = snap_ms(vol, q[None], nq[None], snap_radius, sheet_sigma, min_contrast=-np.inf,
                             clip=clip_density, return_width=True)
        return float(wq[0]) if okq[0] and np.isfinite(wq[0]) and wq[0] > 0 else np.nan

    _, _, w0 = snap_ms(vol, seed[None], n0[None], snap_radius, sheet_sigma, min_contrast=-np.inf,
                       clip=clip_density, return_width=True)
    w0 = float(w0[0]) if np.isfinite(w0[0]) and w0[0] > 0 else np.nan
    max_width = max_width_factor * w0 if snap == "meanshift" and max_width_factor and np.isfinite(w0) else None
    check = {}
    if probes:
        # one snap per probe: the mean-shift stop is global over a batch
        ws = np.array([width_at(q, nq) for q, nq in zip(qp, qn)])
        if snap == "meanshift" and max_width_factor and np.isfinite(ws).any():
            max_width = max_width_factor * float(np.median(ws[np.isfinite(ws)]))
        check = _seed_check(w0, ws[1:], spacing0, sps[1:])
        if log and (check["outlier"] or check["suspect"]):
            log(f"warning: seed unlike the sheet around it ({check['reason'] or check['note']}): on or against a "
                f"foreign body?")
        if seed_check and check["outlier"]:
            raise SeedRejected(f"seed {np.round(seed, 1).tolist()} is unlike the sheet around it ({check['reason']}, "
                               f"{check['probes']} probes within {probes:g} vx): on a foreign body? Move the seed "
                               f"or pass seed_check=False")
    if snap == "meanshift":
        sp, _ = snap_ms(vol, seed[None], n0[None], snap_radius, sheet_sigma, min_contrast=-np.inf, clip=clip_density)
    else:
        sp, _ = _snap(vol, seed[None], n0[None], snap_radius, min_contrast=-np.inf)
    wide = np.arange(-(snap_radius + 3 * sheet_sigma), snap_radius + 3 * sheet_sigma + 1e-9, 0.5)
    pr = np.nan_to_num(vol.sample(sp[0] + wide[:, None] * n0))
    g = np.exp(-0.5 * (wide / sheet_sigma) ** 2)
    contrast0 = float((pr * g).sum() / g.sum() - pr.min())  # the sheet's contrast at the seed
    if min_contrast is None:  # 30% of it, measured the same way snaps are
        min_contrast = 0.3 * contrast0
    prom = basin * contrast0 if basin is not None else None
    if snap == "meanshift":
        def do_snap(p, n, radius, mc, strict=True):
            return snap_ms(vol, p, n, radius, sheet_sigma, min_contrast=mc, clip=clip_density, max_width=max_width,
                           return_width=True, iters=ms_iters, tol=ms_tol, basin=prom, strict=strict)
    else:
        def do_snap(p, n, radius, mc, strict=True):
            return _snap(vol, p, n, radius, min_contrast=mc) + (None,)
    if log:
        log(f"seed normal {np.round(n0, 3)}, sheet spacing {spacing:.2f} vx -> snap radius {snap_radius:.2f} vx, "
            f"sheet sigma {sheet_sigma:.2f} vx ({snap}), min contrast {min_contrast:.1f}")
    if bounds is None:
        sh = vol.shape
        bounds = ((0, 0, 0), (sh[2] - 1, sh[1] - 1, sh[0] - 1))
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)

    N = max_size
    P = np.full((N, N, 3), np.nan)
    c = N // 2
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            P[c + dr, c + dc] = seed + step * (dc * u_ax + dr * v_ax)
    dead = np.zeros((N, N), bool)  # cells that failed to snap: never retried
    ring_age = np.full((N, N), -1)
    ring_age[c - 1:c + 2, c - 1:c + 2] = 0
    WID = np.full((N, N), np.nan)  # bright-run width of each vertex at its birth

    from . import _kernels as K

    box = [c - 1, c + 2, c - 1, c + 2]  # bounding box of valid cells (half-open)

    def window(margin=3):
        return (max(box[0] - margin, 0), min(box[1] + margin, N), max(box[2] - margin, 0), min(box[3] + margin, N))

    def normal_field():
        """Normals on the active window (valid cells: from the grid; others: mean of the nearest valid
        normals, filled ring by ring -- never the seed normal, which is wrong far from the seed)."""
        r0, r1, c0, c1 = window()
        n = Surface(P[r0:r1, c0:c1], (1 / step, 1 / step)).normals().astype(np.float64)
        K.fill_normals(n, 4)
        return n, r0, c0

    def lookup(field, rows, cols):
        n, r0, c0 = field
        out = n[rows - r0, cols - c0]
        bad = ~np.all(np.isfinite(out), 1)
        out[bad] = n0
        return out

    # snap the seed patch
    r, cc_ = np.nonzero(np.isfinite(P[..., 0]))
    P[r, cc_] = do_snap(P[r, cc_], np.tile(n0, (len(r), 1)), snap_radius, min_contrast)[0]

    for g in range(1, generations + 1):
        r0, r1, c0, c1 = window(2)
        sub = np.isfinite(P[r0:r1, c0:c1, 0])
        front = binary_dilation(sub) & ~sub & ~dead[r0:r1, c0:c1]
        fr, fc = np.nonzero(front)
        fr, fc = fr + r0, fc + c0
        keep_ = (fr > 0) & (fr < N - 1) & (fc > 0) & (fc < N - 1)
        fr, fc = fr[keep_], fc[keep_]
        if len(fr) == 0:
            break
        field = normal_field()
        acc = np.zeros((len(fr), 3))
        cnt = np.zeros(len(fr))
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            r1_, c1_ = fr + dr, fc + dc
            r2_, c2_ = fr + 2 * dr, fc + 2 * dc
            okr = (r2_ >= 0) & (r2_ < N) & (c2_ >= 0) & (c2_ < N)
            r1_, c1_, r2_, c2_ = (np.clip(a, 0, N - 1) for a in (r1_, c1_, r2_, c2_))
            p1, p2 = P[r1_, c1_], P[r2_, c2_]
            ok = okr & np.isfinite(p1[:, 0]) & np.isfinite(p2[:, 0])
            if not ok.any():
                continue
            # extrapolate in the tangent plane of p1: a normal-direction error at p1 is carried
            # forward once, not doubled (2*p1 - p2 would turn a 3 vx layer hop into 6 vx)
            n1 = lookup(field, r1_[ok], c1_[ok])
            e = p1[ok] - p2[ok]
            e -= (e * n1).sum(1, keepdims=True) * n1
            if curvature:
                # follow the sheet's turning: rotate the step by the rotation taking n(p2) -> n(p1)
                # (smoothed normals, so a layer hop at p1 is still not amplified)
                n2 = lookup(field, r2_[ok], c2_[ok])
                e = _rotate_like(e, n2, n1)
            acc[ok] += p1[ok] + e
            cnt[ok] += 1
        has = cnt > 0
        fr, fc, pred = fr[has], fc[has], acc[has] / cnt[has, None]
        inb = np.all((pred >= lo) & (pred <= hi), 1)
        dead[fr[~inb], fc[~inb]] = True
        fr, fc, pred = fr[inb], fc[inb], pred[inb]
        if len(fr) == 0:
            break
        nrm = lookup(field, fr, fc)
        if plane_fit:
            pa, pn, pgood = _plane_fit(P, fr, fc, plane_fit)
            sgn = np.sign((pn * nrm).sum(1))
            pn *= np.where(sgn == 0, 1, sgn)[:, None]
            pred = np.where(pgood[:, None], pa, pred)
            nrm = np.where(pgood[:, None], pn, nrm)
        snapped, ok, width = do_snap(pred, nrm, snap_radius, min_contrast)
        if plane_fit and plane_gate is not None:
            with np.errstate(invalid="ignore"):
                ok &= ~(pgood & (np.abs(((snapped - pa) * pn).sum(1)) > plane_gate * spacing))
        if width_rel is not None and width is not None:
            med = np.empty(len(fr))
            K.call(K.window_nanmedian, len(fr) * 25, WID, fr, fc, 2, med)
            ok &= ~(np.isfinite(med) & (width > width_rel * np.maximum(med, 2.0)))
        P[fr[ok], fc[ok]] = snapped[ok]
        if width is not None:
            WID[fr[ok], fc[ok]] = width[ok]
        dead[fr[~ok], fc[~ok]] = True
        ring_age[fr[ok], fc[ok]] = g
        if ok.any():
            box[0] = min(box[0], int(fr[ok].min()))
            box[1] = max(box[1], int(fr[ok].max()) + 1)
            box[2] = min(box[2], int(fc[ok].min()))
            box[3] = max(box[3], int(fc[ok].max()) + 1)
        # relax (springs + straightness) and re-snap the newest rings so extrapolation errors do not accumulate
        for _ in range(relax):
            r0, r1, c0, c1 = window(0)
            band = (ring_age[r0:r1, c0:c1] >= max(g - 2, 1)) & np.isfinite(P[r0:r1, c0:c1, 0])
            br, bc = np.nonzero(band)
            br, bc = br + r0, bc + c0
            if not len(br):
                break
            p0 = P[br, bc]
            tsum = np.zeros((len(br), 3))
            tcnt = np.zeros(len(br))
            # springs: each neighbour proposes the point at distance `step` along the current edge
            # (spacing-preserving, unlike umbrella smoothing which shrinks open boundaries)
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                q = P[np.clip(br + dr, 0, N - 1), np.clip(bc + dc, 0, N - 1)]
                e = p0 - q
                ln = np.linalg.norm(e, axis=1)
                okq = np.isfinite(ln) & (ln > 1e-6)
                tsum[okq] += q[okq] + step * e[okq] / ln[okq, None]
                tcnt[okq] += 1
            # straightness: midpoint of opposite neighbours, only where both exist (unbiased)
            for dr, dc in ((0, 1), (1, 0)):
                qa = P[np.clip(br + dr, 0, N - 1), np.clip(bc + dc, 0, N - 1)]
                qb = P[np.clip(br - dr, 0, N - 1), np.clip(bc - dc, 0, N - 1)]
                okq = np.isfinite(qa[:, 0]) & np.isfinite(qb[:, 0])
                tsum[okq] += 0.5 * (qa[okq] + qb[okq])
                tcnt[okq] += 1
            sm = tcnt >= 1
            newp = p0.copy()
            newp[sm] = 0.5 * p0[sm] + 0.5 * tsum[sm] / tcnt[sm, None]
            nn = lookup(normal_field(), br, bc)
            newp, ok2, _ = do_snap(newp, nn, min(snap_radius, 1.5), min_contrast, strict=not relax_soft)
            P[br[ok2], bc[ok2]] = newp[ok2]
        if log and g % 10 == 0:
            log(f"trace gen {g}: {int(np.isfinite(P[..., 0]).sum())} vertices")
    # crop to the valid bounding box
    valid = np.isfinite(P[..., 0])
    rr, cc = np.nonzero(valid)
    P = P[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    return Surface(P, (1 / step, 1 / step), {"source": "unroller.trace", "seed": seed.tolist(),
                                             "umbilicus": [float(v) for v in umbilicus], "up": up.tolist(),
                                             "snap_radius": snap_radius, "sheet_spacing": spacing,
                                             "sheet_sigma": sheet_sigma, "snap": snap,
                                             "clip_density": clip_density, "max_width": max_width,
                                             "step": step, "seed_check": check,
                                             "safeguards": dict(plane_fit=plane_fit, probes=probes,
                                                                width_rel=width_rel, plane_gate=plane_gate,
                                                                basin=basin, ms_iters=ms_iters, ms_tol=ms_tol,
                                                                relax_soft=relax_soft)})
