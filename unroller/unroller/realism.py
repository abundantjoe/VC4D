"""Measurable phantom realism: the same statistics on real CT and on phantoms, compared as distributions.

Across-sheet statistics come from density profiles along local sheet normals (structure tensor)
at random papyrus voxels:

  thickness_vx     full width at half maximum of the sheet under the probe
  spacing_vx       distance to the nearest neighbouring sheet peak (>= 5 vx away)
  valley_ratio     darkest point between the two sheets / the lower of the two peaks
                   (0 = clean air gap, ~1 = sheets touching / fused)
  peak_intensity   sheet density at the peak
plus the intensity distribution of papyrus voxels and the fraction of zero (clipped air) voxels.

A phantom family is "realistic" for a statistic when its distribution matches the real one; the
report gives medians, IQRs and the 1D Wasserstein distance normalised by the real IQR.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance


def profile_stats(vol, n_probes: int = 300, seed: int = 0, margin: int = 40, reach: float = 32.0) -> dict:
    """``vol``: a Volume/ArrayVolume or a ZYX uint8 array (treated as an ArrayVolume)."""
    from .trace import seed_normal
    from .volume import ArrayVolume

    if isinstance(vol, np.ndarray):
        arr = vol
        vol = ArrayVolume(vol)
    else:
        arr = None
    Z, Y, X = vol.shape
    rng = np.random.default_rng(seed)
    thr = None
    if arr is not None:
        inner = arr[margin:-margin, margin:-margin, margin:-margin]
        thr = np.percentile(inner[inner > 0], 60) if (inner > 0).any() else 1
    out = {k: [] for k in ("thickness_vx", "spacing_vx", "valley_ratio", "peak_intensity")}
    offs = np.arange(-reach, reach + 1e-9, 0.5)
    tries = 0
    while len(out["thickness_vx"]) < n_probes and tries < 20 * n_probes:
        tries += 1
        p = rng.uniform([margin, margin, margin], [X - margin, Y - margin, Z - margin])
        v = float(vol.sample(p[None])[0])
        if thr is not None and v < thr:
            continue
        n = seed_normal(vol, p)
        prof = gaussian_filter(np.nan_to_num(vol.sample(p + offs[:, None] * n)), 2.0)  # 1 vx smoothing
        pk, pr = find_peaks(prof, prominence=0)
        if len(pk) < 2:
            continue
        keep = pk[pr["prominences"] >= 0.15 * pr["prominences"].max()]
        c = keep[np.argmin(np.abs(offs[keep]))]
        if abs(offs[c]) > 6:
            continue
        h = prof[c]
        base = prof.min()
        half = base + 0.5 * (h - base)
        lo = c
        while lo > 0 and prof[lo] > half:
            lo -= 1
        hi = c
        while hi < len(prof) - 1 and prof[hi] > half:
            hi += 1
        others = [k for k in keep if abs(offs[k] - offs[c]) >= 5]
        if not others:
            continue
        nb = min(others, key=lambda k: abs(offs[k] - offs[c]))
        a, b = sorted((c, nb))
        out["thickness_vx"].append((hi - lo) * 0.5)
        out["spacing_vx"].append(abs(offs[nb] - offs[c]))
        out["valley_ratio"].append(float(prof[a:b + 1].min() / max(min(prof[c], prof[nb]), 1e-6)))
        out["peak_intensity"].append(float(h))
    res = {k: np.asarray(v, float) for k, v in out.items()}
    if arr is not None:
        inner = arr[margin:-margin, margin:-margin, margin:-margin]
        res["zero_fraction"] = np.array([float((inner == 0).mean())])
        pap = inner[inner > np.percentile(inner, 50)]
        res["papyrus_intensity"] = pap[np.random.default_rng(seed).integers(0, len(pap), min(20000, len(pap)))].astype(float)
    return res


def compare(real: dict, synth: dict) -> dict:
    rep = {}
    for k in real:
        a, b = real[k], synth.get(k)
        if b is None or len(a) == 0 or len(b) == 0:
            continue
        iqr = np.subtract(*np.percentile(a, [75, 25])) or (abs(np.median(a)) + 1e-9)
        rep[k] = {"real_median": float(np.median(a)), "synth_median": float(np.median(b)),
                  "real_iqr": [float(x) for x in np.percentile(a, [25, 75])],
                  "synth_iqr": [float(x) for x in np.percentile(b, [25, 75])],
                  "w1_over_real_iqr": float(wasserstein_distance(a, b) / iqr)}
    return rep


def pool(stats: list[dict]) -> dict:
    keys = set().union(*[s.keys() for s in stats])
    return {k: np.concatenate([s[k] for s in stats if k in s]) for k in keys}


# ---------------------------------------------------------------------------------------------
# More discriminating statistics (texture, edges, gaps, contacts, spectra, orientation)
# ---------------------------------------------------------------------------------------------
#
# Per probe (random papyrus voxel; sheet normal n from the local gradient structure tensor; two
# tangents: t_circ = n x z-axis (around the roll) and t_ax = n x t_circ (along the axis)):
#
#   core_to_max      profile value at the band's half-max midpoint / band maximum.  ~1 for a sheet
#                    whose core is densest; < 1 when phase-contrast fringes brighten the edges.
#   edge_width_vx    25%-75% rise distance on the sheet flanks (blur, fraying)
#   band_vx, gap_vx  run lengths of material / air along the normal profile, binarised at the
#                    volume's Otsu threshold (only runs bracketed on both sides inside the window)
#   insheet_cv       std/mean of the sheet ridge (max over +-1 vx along n) on a 9x9 vx tangent patch
#   corr_circ_vx     1/e length of the ridge autocorrelation along t_circ (capped at 16)
#   corr_ax_vx       same along t_ax
# Per volume:
#   fused_fraction   fraction of probes whose nearest neighbouring sheet is joined to the probed one
#                    with no sub-threshold voxel between the two peaks (fused contact)
# Per xy slice (every ``slice_step``-th):
#   psd_b0..psd_b3   log10 fraction of the slice's (DC-free) power in radial frequency bands
#                    [0.01,0.04), [0.04,0.08), [0.08,0.16), [0.16,0.5] cycles/vx
# Per papyrus pixel of those slices:
# Per z-block of 24 slices (papyrus voxels, eroded 2 vx, high-pass = volume - gaussian(3 vx)):
#   hp_corr_z1, hp_corr_xy1, hp_corr_xy2   correlation of the high-pass residual between voxels 1 vx
#                    apart along z, and 1 / 2 vx apart in-plane (mean of x and y): the resolution and
#                    noise anisotropy of the reconstruction (real Scroll 1 is much less correlated along z)
#   coherence        2D structure-tensor coherence (lambda1-lambda2)/(lambda1+lambda2), gradient
#                    sigma 1, integration sigma 3: ~1 for clean parallel sheets, lower where sheets
#                    fray, split, merge or are textured.

PSD_BANDS = ((0.01, 0.04), (0.04, 0.08), (0.08, 0.16), (0.16, 0.5))


def otsu(values: np.ndarray) -> float:
    h, e = np.histogram(values, bins=256, range=(0, 256))
    c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    var = w0 * w1 * (m0 - m1) ** 2
    return float(c[np.flatnonzero(var >= var.max() * (1 - 1e-9))].mean())  # centre of a tied plateau


def _runs(b: np.ndarray, val: bool) -> list:
    """Lengths of runs of ``val`` in a 1D boolean array that do not touch either end."""
    d = np.diff(np.r_[False, b == val, False].astype(np.int8))
    st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return [e - s for s, e in zip(st, en) if s > 0 and e < len(b)]


def _acf_len(x: np.ndarray, cap: float) -> float:
    x = x - x.mean()
    v = (x * x).mean()
    if v <= 1e-9:
        return cap
    n = len(x)
    for lag in range(1, n // 2 + 1):
        r = (x[:-lag] * x[lag:]).mean() / v
        if r < 1 / np.e:
            r0 = 1.0 if lag == 1 else (x[:-(lag - 1)] * x[lag - 1:]).mean() / v
            return float(lag - 1 + (r0 - 1 / np.e) / max(r0 - r, 1e-9))
    return cap


def sheet_stats(arr: np.ndarray, n_probes: int = 300, seed: int = 0, margin_xy: int = 36, margin_z: int = 12,
                reach: float = 32.0, step: float = 0.5) -> dict:
    """Per-probe texture/edge/gap statistics of a ZYX uint8 volume (see module notes)."""
    from scipy.ndimage import map_coordinates

    a = np.asarray(arr)
    f = a.astype(np.float32)
    Z, Y, X = a.shape
    rng = np.random.default_rng(seed)
    inner = a[margin_z:Z - margin_z, margin_xy:Y - margin_xy, margin_xy:X - margin_xy]
    thr_bin = otsu(inner.ravel()[rng.integers(0, inner.size, min(inner.size, 400000))])
    thr_probe = np.percentile(inner[inner > 0], 60) if (inner > 0).any() else 1

    def samp(p):  # p (..., 3) xyz
        q = np.asarray(p, float).reshape(-1, 3)
        return map_coordinates(f, [q[:, 2], q[:, 1], q[:, 0]], order=1, mode="nearest").reshape(np.shape(p)[:-1])

    out = {k: [] for k in ("core_to_max", "edge_width_vx", "band_vx", "gap_vx", "insheet_cv",
                           "corr_circ_vx", "corr_ax_vx")}
    fused = []
    offs = np.arange(-reach, reach + 1e-9, step)
    nprb = 0
    tries = 0
    r = 6
    while nprb < n_probes and tries < 30 * n_probes:
        tries += 1
        p = rng.uniform([margin_xy, margin_xy, margin_z], [X - margin_xy, Y - margin_xy, Z - margin_z])
        x, y, z = np.rint(p).astype(int)
        if a[z, y, x] < thr_probe:
            continue
        box = gaussian_filter(f[max(z - r, 0):z + r + 1, y - r:y + r + 1, x - r:x + r + 1], 1.0)
        gz, gy, gx = np.gradient(box)
        g = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
        n = np.linalg.eigh(g.T @ g)[1][:, -1]
        tc = np.cross(n, [0.0, 0.0, 1.0])
        if np.linalg.norm(tc) < 0.3:  # sheet nearly perpendicular to the axis: skip
            continue
        tc /= np.linalg.norm(tc)
        ta = np.cross(n, tc)
        prof = gaussian_filter(samp(p + offs[:, None] * n), 1.0)  # 0.5 vx smoothing
        pk, pr = find_peaks(prof, prominence=0)
        if len(pk) < 2:
            continue
        keep = pk[pr["prominences"] >= 0.15 * pr["prominences"].max()]
        c = keep[np.argmin(np.abs(offs[keep]))]
        if abs(offs[c]) > 6:
            continue
        nprb += 1
        # band edges from the binarised profile around the peak
        b = prof >= thr_bin
        lo, hi = c, c
        while lo > 0 and b[lo - 1]:
            lo -= 1
        while hi < len(prof) - 1 and b[hi + 1]:
            hi += 1
        mx = prof[lo:hi + 1].max()
        mid = (lo + hi) // 2
        out["core_to_max"].append(float(prof[mid] / max(mx, 1e-6)))
        # 25-75% rise on each flank, relative to the local floor
        for side in (-1, 1):
            j = c
            base = prof[: c + 1].min() if side < 0 else prof[c:].min()
            l25, l75 = base + 0.25 * (prof[c] - base), base + 0.75 * (prof[c] - base)
            j75 = None
            while 0 < j < len(prof) - 1:
                if j75 is None and prof[j] < l75:
                    j75 = j
                if prof[j] < l25:
                    break
                j += side
            if j75 is not None and 0 < j < len(prof) - 1:
                out["edge_width_vx"].append(abs(j - j75) * step)
        bs = b[::int(round(1 / step))]
        out["band_vx"] += [float(v) for v in _runs(bs, True)]
        out["gap_vx"] += [float(v) for v in _runs(bs, False)]
        others = [k for k in keep if abs(offs[k] - offs[c]) >= 5]
        if others:
            nb = min(others, key=lambda k: abs(offs[k] - offs[c]))
            i0, i1 = sorted((c, nb))
            fused.append(bool(prof[i0:i1 + 1].min() >= thr_bin))
        # ridge texture on the tangent plane at the peak
        pc = p + offs[c] * n
        u = np.arange(-4, 5, 1.0)
        dn = np.array([-1.0, 0.0, 1.0])
        pts = pc + u[:, None, None, None] * tc + u[None, :, None, None] * ta + dn[None, None, :, None] * n
        patch = samp(pts).max(-1)
        out["insheet_cv"].append(float(patch.std() / max(patch.mean(), 1e-6)))
        L = np.arange(-16, 17, 1.0)
        for key, t in (("corr_circ_vx", tc), ("corr_ax_vx", ta)):
            line = samp(pc + L[:, None, None] * t + dn[None, :, None] * n).max(-1)
            out[key].append(_acf_len(line, 16.0))
    res = {k: np.asarray(v, float) for k, v in out.items()}
    res["fused_fraction"] = np.array([float(np.mean(fused)) if fused else 0.0])
    return res


def slice_stats(arr: np.ndarray, slice_step: int = 8, margin: int = 8, n_coh: int = 3000, seed: int = 0) -> dict:
    """Radially averaged power-spectrum band fractions and structure-tensor coherence of xy slices."""
    a = np.asarray(arr)
    Z, Y, X = a.shape
    rng = np.random.default_rng(seed)
    sl = a[margin:Z - margin:slice_step].astype(np.float32)
    ny, nx = sl.shape[1:]
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.rfftfreq(nx)[None, :]
    fr = np.hypot(fy, fx)
    win = (np.hanning(ny)[:, None] * np.hanning(nx)[None, :]).astype(np.float32)
    thr = otsu(a.ravel()[rng.integers(0, a.size, min(a.size, 400000))])
    res = {f"psd_b{i}": [] for i in range(len(PSD_BANDS))}
    coh = []
    for s in sl:
        P = np.abs(np.fft.rfft2((s - s.mean()) * win)) ** 2
        tot = P[(fr >= PSD_BANDS[0][0]) & (fr <= 0.5)].sum() + 1e-12
        for i, (lo, hi) in enumerate(PSD_BANDS):
            res[f"psd_b{i}"].append(float(np.log10(P[(fr >= lo) & (fr < hi if i < len(PSD_BANDS) - 1 else fr <= hi)].sum() / tot + 1e-12)))
        gy = gaussian_filter(s, 1.0, order=(1, 0))
        gx = gaussian_filter(s, 1.0, order=(0, 1))
        jxx, jyy, jxy = (gaussian_filter(v, 3.0) for v in (gx * gx, gy * gy, gx * gy))
        tr = jxx + jyy
        disc = np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy)
        c = disc / np.maximum(tr, 1e-6)
        m = (s >= thr)[8:-8, 8:-8]
        cv = c[8:-8, 8:-8][m]
        if cv.size:
            coh.append(cv[rng.integers(0, cv.size, min(cv.size, max(1, n_coh // len(sl))))])
    out = {k: np.asarray(v, float) for k, v in res.items()}
    out["coherence"] = np.concatenate(coh).astype(float) if coh else np.zeros(0)
    return out


def anisotropy_stats(arr: np.ndarray, block: int = 24, thr: float = 100.0) -> dict:
    """High-pass autocorrelation of papyrus voxels along z and in-plane, per z-block (see notes)."""
    from scipy.ndimage import binary_erosion

    f = np.asarray(arr).astype(np.float32)
    hp = f - gaussian_filter(f, 3.0)
    m = binary_erosion(f > thr, iterations=2)
    out = {"hp_corr_z1": [], "hp_corr_xy1": [], "hp_corr_xy2": []}

    def corr(h, mm, ax, lag):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax], b[ax] = slice(0, -lag), slice(lag, None)
        k = mm[tuple(a)] & mm[tuple(b)]
        if k.sum() < 100:
            return np.nan
        return float(np.corrcoef(h[tuple(a)][k], h[tuple(b)][k])[0, 1])

    for z0 in range(0, f.shape[0] - block + 1, block):
        h, mm = hp[z0:z0 + block], m[z0:z0 + block]
        out["hp_corr_z1"].append(corr(h, mm, 0, 1))
        out["hp_corr_xy1"].append(0.5 * (corr(h, mm, 1, 1) + corr(h, mm, 2, 1)))
        out["hp_corr_xy2"].append(0.5 * (corr(h, mm, 1, 2) + corr(h, mm, 2, 2)))
    return {k: np.asarray([x for x in v if np.isfinite(x)], float) for k, v in out.items()}


def _normal_at(f, p, r=8, sigma=2.0):
    """Sheet normal from the gradient structure tensor of a smoothed box (sigma 2 vx: real Scroll 1 noise is
    nearly independent between slices, and 1 vx gradients are biased towards z by it)."""
    x, y, z = np.rint(p).astype(int)
    box = gaussian_filter(f[max(z - r, 0):z + r + 1, max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1], sigma)
    gz, gy, gx = np.gradient(box)
    g = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    return np.linalg.eigh(g.T @ g)[1][:, -1]


def geometry_stats(arr: np.ndarray, n_probes: int = 200, seed: int = 0, reach: float = 16.0) -> dict:
    """Sheet geometry: bending, kinks, lamina splits, splits/merges along z, sheet count.

    Per probe (random papyrus voxel p, normal n, tangents t_circ = n x z, axis direction z):
      turn_z8_deg, turn_z32_deg     angle between the sheet normals at p and 8 / 32 vx further along z
      turn_s8_deg, turn_s32_deg     same along the sheet around the roll (t = n x z); both directions march
                                    in 8 vx steps, each step re-centred on the brightest voxel within +-3 vx
                                    along the current normal, so the walk stays on the probed sheet
      lamina_split (-> lamina_split_frac per volume)  1 if the band under p (binarised at Otsu) holds an internal dip below 80 %
                                    of the lower of its two flanking maxima (two or more laminae), else 0
      band_change_z8 (-> band_change_z8_frac per volume)  1 if the number of bands crossing +-``reach`` vx along n differs between p
                                    and p + 8 vx along z (a split, merge or termination), else 0
      normal_z_abs                  |z-component| of the sheet normal at every papyrus probe (orientation: a
                                    volume with axes swapped, or sheets lying across the axis, fails it)
    Per volume:
      kink_frac                     fraction of probes whose normal turns > 30 deg within 8 vx (z or s)
      bands_per_100vx               mean number of bands per 100 vx of a 64 vx normal profile (sparse crops)
    """
    from scipy.ndimage import map_coordinates

    a = np.asarray(arr)
    f = a.astype(np.float32)
    Z, Y, X = a.shape
    rng = np.random.default_rng(seed + 7)
    m = 12
    inner = a[m:Z - m, m:Y - m, m:X - m]
    thr = otsu(inner.ravel()[rng.integers(0, inner.size, min(inner.size, 400000))])

    def samp(q):
        q = np.asarray(q, float).reshape(-1, 3)
        return map_coordinates(f, [q[:, 2], q[:, 1], q[:, 0]], order=1, mode="nearest")

    def ang(n1, n2):
        return float(np.degrees(np.arccos(min(abs(float(n1 @ n2)), 1.0))))

    offs = np.arange(-reach, reach + 1e-9, 1.0)
    out = {k: [] for k in ("turn_z8_deg", "turn_z32_deg", "turn_s8_deg", "turn_s32_deg", "lamina_split", "band_change_z8",
                           "normal_z_abs")}
    kinks, nbands = [], []
    lo_xy = 28
    tries = 0
    while len(out["lamina_split"]) < n_probes and tries < 40 * n_probes:
        tries += 1
        p = rng.uniform([lo_xy, lo_xy, 8], [X - lo_xy, Y - lo_xy, max(Z - 40, 9)])
        x, y, z = np.rint(p).astype(int)
        if a[z, y, x] < thr:
            continue
        n = _normal_at(f, p)
        out["normal_z_abs"].append(float(abs(n[2])))  # orientation: sheets of a roll contain the axis (|n_z| ~ 0)
        tc = np.cross(n, [0.0, 0.0, 1.0])
        if np.linalg.norm(tc) < 0.3:
            continue
        tc /= np.linalg.norm(tc)
        prof = gaussian_filter(samp(p + offs[:, None] * n), 0.7)
        b = prof >= thr
        c = int(np.argmin(np.abs(offs)))
        if not b[c]:
            continue
        lo, hi = c, c
        while lo > 0 and b[lo - 1]:
            lo -= 1
        while hi < len(b) - 1 and b[hi + 1]:
            hi += 1
        split = 0
        if hi - lo >= 5 and lo > 0 and hi < len(b) - 1:
            seg = prof[lo:hi + 1]
            k = int(np.argmin(seg[1:-1])) + 1
            if seg[k] < 0.8 * min(seg[:k].max(), seg[k + 1:].max()):
                split = 1
        out["lamina_split"].append(float(split))
        runs = len(_runs(np.r_[False, b, False], True))
        long = gaussian_filter(samp(p + np.arange(-32, 32.5, 1.0)[:, None] * n), 0.7) >= thr
        nbands.append(float(np.sum(long[1:] & ~long[:-1])) / (len(long) - 1) * 100)  # band starts per 100 vx
        q = p + [0, 0, 8.0]
        b8 = gaussian_filter(samp(q + offs[:, None] * n), 0.7) >= thr
        out["band_change_z8"].append(float(len(_runs(np.r_[False, b8, False], True)) != runs))
        turns = {}
        for axis in ("z", "s"):  # march along the sheet in 8 vx steps, re-centring on it each step
            q, nq = p.copy(), n
            for i in range(1, 5):
                if axis == "z":
                    d = np.array([0.0, 0.0, 8.0])
                else:
                    t = np.cross(nq, [0.0, 0.0, 1.0])
                    if np.linalg.norm(t) < 0.3:
                        break
                    d = 8 * t / np.linalg.norm(t) * np.sign(t @ tc)
                cand = q + d + np.arange(-3, 3.5, 0.5)[:, None] * nq
                q = cand[int(np.argmax(samp(cand)))]
                if not (6 <= q[0] < X - 6 and 6 <= q[1] < Y - 6 and 6 <= q[2] < Z - 6):
                    break
                nq = _normal_at(f, q)
                if i in (1, 4):
                    key = f"{axis}{8 * i}"
                    turns[key] = ang(n, nq)
                    out[f"turn_{key}_deg"].append(turns[key])
        kinks.append(float(max(turns.get("z8", 0), turns.get("s8", 0)) > 30))
    res = {k: np.asarray(v, float) for k, v in out.items() if not k.startswith(("lamina", "band_change"))}
    for k in ("lamina_split", "band_change_z8"):  # binary per probe -> one fraction per volume
        res[k + "_frac"] = np.array([float(np.mean(out[k])) if out[k] else 0.0])
    res["kink_frac"] = np.array([float(np.mean(kinks)) if kinks else 0.0])
    res["bands_per_100vx"] = np.array([float(np.mean(nbands)) if nbands else 0.0])
    return res


def realism_stats(arr: np.ndarray, n_probes: int = 300, seed: int = 0) -> dict:
    """All statistics of a ZYX uint8 volume: profile, sheet, slice, anisotropy and geometry statistics."""
    res = profile_stats(arr, n_probes=n_probes, seed=seed, margin=min(40, (min(np.shape(arr)) - 2) // 2 - 1))
    res.update(sheet_stats(arr, n_probes=n_probes, seed=seed))
    res.update(slice_stats(arr, seed=seed))
    res.update(anisotropy_stats(arr))
    res.update(geometry_stats(arr, seed=seed))
    return res


def summary_table(rep: dict) -> str:
    """Plain-text table of a ``compare`` report."""
    lines = [f"{'statistic':18s} {'real med':>9s} {'synth med':>9s} {'W1/IQR':>7s}"]
    for k in sorted(rep):
        r = rep[k]
        lines.append(f"{k:18s} {r['real_median']:9.3f} {r['synth_median']:9.3f} {r['w1_over_real_iqr']:7.2f}")
    return "\n".join(lines)
