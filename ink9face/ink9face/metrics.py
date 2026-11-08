"""Ink-map metrics in physical units (defined at 1 px = 1 voxel; constants assume ~9 um voxels).

score(p, mask, lab) returns
  auc       pixel ROC AUC (Mann-Whitney on ranks) inside mask
  auc_null  auc minus the mean AUC against the label rolled by 12 large shifts (128-512 px): how much of the AUC is
            spatially specific rather than a global brightness bias
  auc_tol   AUC ignoring a +-tol px band around label edges (label edges are uncertain by a few voxels)
  auc_37um  AUC on a 4x4 px block grid (37 um at 9.362 um), comparable across scan resolutions
  letter    letter-scale correlation: band-pass (DoG sigma 4 -> 24 px) correlation between map and label inside the
            48-px-eroded mask, minus the same correlation against shifted labels (a shift null)
Bootstrap helpers resample MESHES (segments), not windows, because two windows of one mesh are not independent.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata, binomtest

SHIFTS = [(0, s) for s in (128, -128, 256, -256, 512, -512)] + [(s, 0) for s in (128, -128, 256, -256, 512, -512)]


def auc_ranks(r, y):
    n1 = int(y.sum()); n0 = y.size - n1
    if n1 < 50 or n0 < 50:
        return float("nan")
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc(p, y):
    return auc_ranks(rankdata(p), y)


def band(a, m, s1=4.0, s2=24.0):
    a = np.where(m, a, 0).astype(np.float32); w = m.astype(np.float32)
    g = ndi.gaussian_filter
    return g(a, s1) / np.maximum(g(w, s1), 1e-3) - g(a, s2) / np.maximum(g(w, s2), 1e-3)


def letter_r(h, lab, m, erode=48):
    me = ndi.binary_erosion(m, iterations=erode)
    if me.sum() < 20000:
        return float("nan")
    a = band(h, me); b = band(lab.astype(np.float32), me)
    c = lambda x, y, mm: float(np.corrcoef(x[mm], y[mm])[0, 1])
    return c(a, b, me) - float(np.nanmean([c(a, np.roll(b, s, (0, 1)), me & np.roll(me, s, (0, 1))) for s in SHIFTS]))


def score(p, mask, lab, tol=4, blk=4):
    p = np.asarray(p, np.float32); lab = lab.astype(bool); mask = mask.astype(bool)
    r = rankdata(p[mask]); a0 = auc_ranks(r, lab[mask])
    nl = float(np.nanmean([auc_ranks(r, np.roll(lab, s, (0, 1))[mask]) for s in SHIFTS]))
    edge = ndi.binary_dilation(lab, iterations=tol) & ~ndi.binary_erosion(lab, iterations=tol)
    mt = mask & ~edge
    at = auc_ranks(rankdata(p[mt]), lab[mt]) if mt.sum() > 1000 else float("nan")
    H, W = (mask.shape[0] // blk) * blk, (mask.shape[1] // blk) * blk
    B = lambda x: x[:H, :W].reshape(H // blk, blk, W // blk, blk).mean((1, 3))
    mb = B(mask.astype(np.float32)) == 1
    pb = B(np.where(mask, p, 0)); lb = B(lab.astype(np.float32)) > 0.5
    ab = auc_ranks(rankdata(pb[mb]), lb[mb]) if mb.sum() > 500 else float("nan")
    return dict(auc=a0, auc_null=a0 - nl, auc_tol=at, auc_37um=ab, letter=letter_r(p, lab, mask))


def letter_band_energy(p, m, erode=48):
    """LABEL-FREE: std of the letter-scale band-pass of logit(p) inside the eroded mask."""
    pc = np.clip(p, 1e-3, 1 - 1e-3); lg = np.log(pc / (1 - pc))
    me = ndi.binary_erosion(m, iterations=erode)
    if me.sum() < 1000:
        me = m
    return float(band(lg, me)[me].std())


def mesh_paired(diffs, meshes, n_boot=10000, seed=0):
    """diffs: per-window paired differences; meshes: mesh id per window.  Averages windows within a mesh, then
    bootstraps meshes.  Returns mean, 95 % CI, meshes better / total, one-sided sign-test p."""
    diffs = np.asarray(diffs, float); meshes = np.asarray(meshes)
    ok = np.isfinite(diffs); diffs, meshes = diffs[ok], meshes[ok]
    ids = sorted(set(meshes.tolist()))
    if not ids:
        return dict(n_meshes=0)
    D = np.array([diffs[meshes == i].mean() for i in ids])
    rng = np.random.default_rng(seed)
    bs = D[rng.integers(0, len(D), (n_boot, len(D)))].mean(1)
    k = int((D > 0).sum()); n = int((D != 0).sum())
    return dict(mean=float(D.mean()), ci=[float(x) for x in np.percentile(bs, [2.5, 97.5])], better=k, n_meshes=len(D),
                n_windows=int(ok.sum()), sign_p=float(binomtest(k, n, 0.5, alternative="greater").pvalue) if n else float("nan"))
