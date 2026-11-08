"""Which face of the sheet (= which layer order) to feed a 9 um ink model.

Ink models with a 3D stem take a stack of layers rendered along the surface normal, and they are NOT symmetric in
depth: fed the stack in the wrong order they read the other face of the sheet (AUC near chance on public labels).
villa's ink inference (`koine_machines/inference/infer.py`) takes the order from `--direction forward|reverse|both`
and defaults to `forward` (the native order); the inference container reverses only with FORCE_REVERSE=true.
There is no rule in the inference path (villa_pr/ adds `--direction auto` / `--reverse_layers auto` with this one).

Conventions used here
  +n         VC3D normal of the tifxyz grid, n = normalize(dP/dcol x dP/drow) (tifxyz.vc_normals).
  native     stack whose layer index increases along +n (what vc_render_tifxyz produces).
  reversed   native[::-1] (layer index increases along -n).

Rules (all label-free)
  anti_normal   feed `reversed`.  Measured on organizer meshes of 5 public scrolls (README): right on 41/43 windows
                against clean-model truth (PHerc0500P2, PHerc0814) and 97/99 against the TRACE face oracle (all 5).
  native        feed `native` (the do-nothing default of a renderer).
  toward_axis   feed the order whose index increases toward the scroll axis (our earlier pipeline; meaningless for
                flat fragments).
  band          two-face model check: run the model on both orders, keep the one with more letter-scale band-pass
                energy of logit(p) (metrics.letter_band_energy).  32/43 per window; as a scroll-level vote it was right on 2/2 scrolls.
"""
import numpy as np

from .metrics import letter_band_energy

RULES = ("anti_normal", "native", "toward_axis", "band", "band_scroll_vote")


def orient(stack_native, rule="anti_normal"):
    """Model input from a native-order stack (layers along +n).  Only the fixed rules are allowed here."""
    if rule == "anti_normal":
        return stack_native[::-1]
    if rule == "native":
        return stack_native
    raise ValueError("orient() takes 'anti_normal' or 'native'; data-dependent rules need both maps (choose_band)")


def toward_axis_is_native(points, normals, axis_xy):
    """True if +n points toward the scroll axis (vertical line through axis_xy) for most valid nodes."""
    ok = np.all(np.isfinite(points), -1) & np.all(np.isfinite(normals), -1)
    d = (normals[ok][:, :2] * (np.asarray(axis_xy)[None] - points[ok][:, :2])).sum(1)
    return bool(np.median(d) > 0)


def choose_band(p_native, p_reversed, mask):
    """Two-face model check for one window: returns ('reversed' | 'native', energy_native, energy_reversed)."""
    en, er = letter_band_energy(p_native, mask), letter_band_energy(p_reversed, mask)
    return ("reversed" if er > en else "native"), en, er


def scroll_vote(pairs):
    """Scroll-level band vote: pairs = [(energy_native, energy_reversed), ...] over windows of one scroll.
    Returns 'reversed' if the summed log-ratio favours reversed."""
    s = float(np.sum([np.log(er / en) for en, er in pairs if en > 0 and er > 0]))
    return ("reversed" if s > 0 else "native"), s
