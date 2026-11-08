"""Label-free depth registration between two mesh conventions.

Two tracers can put "layer 0" at different depths through the same sheet (centre of the sheet vs its surface,
or a few voxels off).  A model trained on one convention loses accuracy on the other.  The mean CT profile across the
rendered layers, robust-normalised per window, is a label-free fingerprint of the convention: the integer shift that
best aligns the mean profile of a target set of windows with a reference set (e.g. the organizers' benchmark meshes,
read on their correct face) is the depth offset to apply.  Spread is reported from per-window best shifts.
"""
import numpy as np


def profile(stack, valid):
    """Robust-z mean intensity per layer: stack (L, H, W) uint8/float, valid (H, W) bool -> (L,)"""
    x = stack[:, valid].astype(np.float32)
    med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826 + 1e-3
    return ((x - med) / mad).mean(1)


def best_shift(ref, tgt, max_shift=6, core=(6, 19)):
    """Shift s maximising corr(ref[core], roll(tgt, -s)[core]): target layer index + s matches the reference."""
    a, b = core
    sc = {s: float(np.corrcoef(ref[a:b], np.roll(tgt, -s)[a:b])[0, 1]) for s in range(-max_shift, max_shift + 1)}
    return max(sc, key=sc.get), sc


def compare(ref_profiles, tgt_profiles, max_shift=6):
    R, T = np.asarray(ref_profiles), np.asarray(tgt_profiles)
    s, sc = best_shift(R.mean(0), T.mean(0), max_shift)
    per = np.array([best_shift(R.mean(0), t, max_shift)[0] for t in T])
    return dict(shift=int(s), corr=sc, per_window_median=float(np.median(per)), per_window_iqr=np.percentile(per, [25, 75]).tolist(),
                n_ref=len(R), n_tgt=len(T))
