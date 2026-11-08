"""Ink-side fine-valley texture: a transparent ink cue for flattened layer stacks, plus an overlay.

(The module and CLI keep the name ``crackle`` for compatibility; what they measure is fine valley texture on the
ink side of the surface, not a verified crack network.)

Carbon ink on Herculaneum papyrus has almost the same X-ray density as the papyrus, so a flattened
render barely shows it as brightness.  The design hypothesis was *crackle*: a thin film (~3-20 um,
0.4-2.5 vx at 7.91 um) lying on the writing surface, dried and cracked into a polygonal "mud-crack"
network, slightly raised above the papyrus fibers, and hiding the fiber texture under it.
Letters are 2-4 mm (250-500 px), strokes ~0.3-0.5 mm (40-60 px).

The detector turns those statements into five measurable maps, each computed on a layer stack
(n, H, W) whose surface layer is ``surface`` (VC renders: layer 32 of 00..64; unroller renders: the
middle layer) and whose ink face is on side ``ink_side`` of it (+1 = higher layer indices):

  crack_energy    (a) density of thin dark valleys (Hessian, sigma ~1.5 px) in the layers 3..7 vx
                      above the surface, pooled at stroke scale, as a log-ratio to the local background;
  crack_isotropy  (a) how isotropic those valleys are (1 - coherence of their doubled-angle
                      orientations): a crack network has no preferred direction, papyrus fibers do;
  coherence_loss  (b) loss of fiber anisotropy on the surface (the ``fibers.orientation_field``
                      Hessian estimator, mask-aware) relative to the local background;
  relief          (c) brightening of the crackle band above the surface relative to the layers
                      below it (raised material), relative to the local background;
  smoothness      (d) loss of fine texture on the surface itself, log-ratio to the background.

What the labelled Scroll 1 segments say (scripts/crackle_eval.py, README "Ink-side fine-valley texture"):
only (a) crack_energy carries a transferable signal, and it is weak.  It is not clearly better than
trivial label-free cues: distance to the mask edge alone, or a nuisance model of brightness, texture,
sheet offset, fiber orientation and edge distance, match or beat it on one held-out segment.  The other
hypotheses do not hold (valleys over labelled ink are *more* oriented, the surface there is *rougher*).
The score also fires on tear and fold edges and fiber-bundle crossings.  Treat it as a hint for where
to look, not as ink.

Every map is robust-z-scored inside the valid area of its own segment (median / MAD), so a model
fitted on one segment can be applied to another rendered with a different contrast.  The maps are
combined by a small logistic regression (``CrackleModel``; one weight per map) fitted on a training
segment only.  ``default_model()`` was fitted on segment 20230531121653 (Scroll 1, official VC layers).
Scores within ``edge_margin`` px of the valid-area edge are suppressed (NaN): the mask edge is where
segments tear and leave the sheet, and there the score was enriched for reasons unrelated to ink.
The absolute ``threshold`` does not transfer between segments; the operating point is a per-segment
percentile (``CrackleModel.percentile_threshold``: flag the same fraction as on the training segment).

Vectorised scipy only: all five maps of a 2560 x 2560 x 33 stack take 16 s on 4 shared CPUs (crackle_eval --bench).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, uniform_filter

FEATURES = ("crack_energy", "crack_isotropy", "coherence_loss", "relief", "smoothness")


# --------------------------------------------------------------------------------------- filters
def blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; for sigma > 4 a 3-pass box cascade (same variance, cost independent of sigma)."""
    x = np.asarray(x, np.float32)
    if sigma <= 0:
        return x
    if sigma <= 4:
        return gaussian_filter(x, sigma, mode="reflect")
    # 3 odd boxes of widths wl, wl+2 mixed so the summed variance sum (w^2 - 1) / 12 is sigma^2 (Kovesi 2010)
    wl = int(np.floor(np.sqrt(4.0 * sigma * sigma + 1.0)))
    wl -= 1 - wl % 2
    m = int(round((12 * sigma * sigma - 3 * wl * wl - 12 * wl - 9) / (-4 * wl - 4)))
    for i in range(3):
        x = uniform_filter(x, wl if i < m else wl + 2, mode="reflect")
    return x


def masked_blur(x: np.ndarray, valid: np.ndarray, sigma: float, min_weight: float = 1e-3) -> np.ndarray:
    """Normalised convolution: blur of x over valid (and finite) pixels only; NaN where (almost)
    nothing is valid."""
    ok = valid & np.isfinite(x)
    v = ok.astype(np.float32)
    num = blur(np.where(ok, x, 0).astype(np.float32), sigma)
    den = blur(v, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < min_weight] = np.nan
    return out


def hessian_eig(img: np.ndarray, sigma: float):
    """Scale-normalised Hessian eigenvalues l1 >= l2 and the doubled angle of the l1 eigenvector."""
    f = np.asarray(img, np.float32)
    hyy = gaussian_filter(f, sigma, order=(2, 0))
    hxx = gaussian_filter(f, sigma, order=(0, 2))
    hxy = gaussian_filter(f, sigma, order=(1, 1))
    s2 = sigma * sigma
    tr = (hxx + hyy) * s2
    dxy = (hxx - hyy) * s2
    oxy = 2 * hxy * s2
    disc = np.sqrt(dxy * dxy + oxy * oxy)
    return 0.5 * (tr + disc), 0.5 * (tr - disc), np.arctan2(oxy, dxy)


# --------------------------------------------------------------------------------------- features
@dataclass(frozen=True)
class CrackleParams:
    """Geometry and scales of the detector (pixels / voxels).  Defaults: Scroll 1 at 7.91 um."""
    surface: int | None = None        # surface layer index in the stack; None = middle layer
    ink_side: int = 1                 # +1: the ink face is towards higher layer indices
    step: float = 1.0                 # voxels between layers
    band: tuple = (3.0, 7.0)          # crackle band: voxels from the surface on the ink side
    ref_band: tuple = (-6.0, -2.0)    # reference band for relief (same sign convention)
    surface_band: tuple = (-1.0, 1.0)  # layers averaged for the surface texture (smoothness)
    fiber_band: tuple = (-1.0, 1.0)   # layers averaged for the fiber coherence
    sigma_crack: float = 1.5          # Hessian scale of crack valleys (cracks are a few px wide)
    sigma_pool: float = 16.0          # pooling scale (FWHM 38 px ~ one stroke width, 40-60 px)
    sigma_bg: float = 192.0           # local background (> letters, 250-500 px); <= 0: segment median
    fiber_width: float = 2.0          # Hessian scale of the fiber coherence (as fibers.orientation_field)
    sigma_fine: tuple = (0.7, 2.0)    # DoG band of the fine surface texture
    edge_margin: float = 64.0         # scores within this many px of the valid-area edge are suppressed (0: off)

    def layers(self, n: int, band) -> list[int]:
        s = (n - 1) // 2 if self.surface is None else self.surface
        lo, hi = band
        ks = sorted({s + self.ink_side * int(round(o / self.step))
                     for o in np.arange(lo, hi + 1e-9, self.step)})
        bad = [k for k in ks if not 0 <= k < n]
        if bad:
            raise ValueError(f"band {band} needs layers {bad} outside the stack (n={n}, surface={s})")
        return ks


def _robust_scale(img, valid):
    v = img[valid & np.isfinite(img)]
    if v.size == 0:
        return 0.0, 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    return med, (mad if mad > 0 else float(v.std()) or 1.0)


def crackle_features(stack, valid=None, params: CrackleParams = CrackleParams(), features=FEATURES) -> dict:
    """The raw feature maps (H, W) float32 of a layer stack (n, H, W); NaN outside ``valid``.

    ``stack`` may be any numeric dtype and contain NaN (invalid pixels).  ``valid`` (H, W) bool,
    default: pixels finite in every layer used.  Layers are read lazily, so a memory-mapped stack works.
    """
    p = params
    n = len(stack)
    band = p.layers(n, p.band)
    ref = p.layers(n, p.ref_band)
    surf = p.layers(n, p.surface_band)
    fib = p.layers(n, p.fiber_band)
    s0 = (n - 1) // 2 if p.surface is None else p.surface

    def lay(k):
        return np.asarray(stack[k], np.float32)

    first = lay(s0)
    if valid is None:
        valid = np.ones(first.shape, bool)
    valid = valid.copy()
    for k in sorted(set(band) | set(ref) | set(surf) | set(fib)):
        valid &= np.isfinite(lay(k))
    # one global intensity scale (from the surface layer) keeps the depth profile intact for relief
    med, sc = _robust_scale(first, valid)
    # derivative filters see the zero fill outside ``valid``: their responses are pooled only from
    # pixels at least 3 sigma inside (normalised convolution extrapolates the rest of the valid area)
    dist = distance_transform_edt(valid) if not valid.all() else None

    def inner(sigma):
        return valid if dist is None else dist > 3.0 * sigma

    def norm(k):
        a = (lay(k) - med) / sc
        a[~valid] = 0.0
        return a

    out = {}
    pool, bg = p.sigma_pool, p.sigma_bg
    eps = 1e-3

    def background(x):  # local background at sigma_bg; sigma_bg <= 0: the segment median
        if bg > 0:
            return masked_blur(x, valid, bg)
        v = x[valid & np.isfinite(x)]
        return np.float32(np.median(v)) if v.size else np.float32(0)

    if "crack_energy" in features or "crack_isotropy" in features:
        V = np.zeros(first.shape, np.float32)
        C = np.zeros(first.shape, np.float32)
        S = np.zeros(first.shape, np.float32)
        for k in band:
            l1, _, ang2 = hessian_eig(norm(k), p.sigma_crack)
            v = np.maximum(l1, 0)  # dark valley: large positive curvature across it
            V += v
            C += v * np.cos(ang2)
            S += v * np.sin(ang2)
        V /= len(band)
        w = inner(p.sigma_crack)
        Vp = masked_blur(V, w, pool)
        if "crack_energy" in features:
            out["crack_energy"] = np.log(Vp + eps) - np.log(background(Vp) + eps)
        if "crack_isotropy" in features:
            with np.errstate(invalid="ignore", divide="ignore"):
                iso = 1.0 - np.hypot(masked_blur(C / len(band), w, pool),
                                     masked_blur(S / len(band), w, pool)) / (Vp + eps)
            out["crack_isotropy"] = iso - background(iso)
        del V, C, S

    if "coherence_loss" in features:
        # fibers.orientation_field (method="hessian") with mask-aware pooling: doubled-angle mean of
        # the axis of largest |curvature|, weighted by the anisotropy l1 - l2
        comp = np.mean([norm(k) for k in fib], 0).astype(np.float32)
        l1, l2, ang2 = hessian_eig(comp, max(1.05, p.fiber_width))
        aniso = l1 - l2
        ang2 = np.where(np.abs(l2) > np.abs(l1), ang2 + np.pi, ang2)
        w = inner(p.fiber_width)
        with np.errstate(invalid="ignore", divide="ignore"):
            coh = np.hypot(masked_blur(aniso * np.cos(ang2), w, pool),
                           masked_blur(aniso * np.sin(ang2), w, pool)) / masked_blur(aniso, w, pool)
        out["coherence_loss"] = background(coh) - coh
    if "relief" in features:
        a = np.mean([norm(k) for k in band], 0)
        b = np.mean([norm(k) for k in ref], 0)
        r = masked_blur(a - b, valid, pool)
        out["relief"] = r - background(r)
    if "smoothness" in features:
        comp = np.mean([norm(k) for k in surf], 0).astype(np.float32)
        d = blur(comp, p.sigma_fine[0]) - blur(comp, p.sigma_fine[1])
        e = masked_blur(d * d, inner(p.sigma_fine[1]), pool)
        out["smoothness"] = np.log(background(e) + eps) - np.log(e + eps)

    for k in out:
        out[k] = np.asarray(out[k], np.float32)
        out[k][~valid] = np.nan
    out["valid"] = valid
    return out


def edge_distance(valid: np.ndarray) -> np.ndarray:
    """Distance (px) of every pixel to the nearest invalid pixel or the image border (0 outside ``valid``)."""
    return distance_transform_edt(np.pad(np.asarray(valid, bool), 1))[1:-1, 1:-1].astype(np.float32)


def standardize(feats: dict, names=FEATURES, valid=None) -> np.ndarray:
    """Robust z-score of each feature map inside the valid area of its own segment -> (k, H, W)."""
    valid = feats["valid"] if valid is None else valid
    Z = []
    for nm in names:
        f = feats[nm]
        med, sc = _robust_scale(f, valid)
        Z.append(((f - med) / sc).astype(np.float32))
    return np.stack(Z)


# --------------------------------------------------------------------------------------- model
@dataclass
class CrackleModel:
    """Logistic regression on standardised feature maps: p = sigmoid(bias + sum_i w_i z_i)."""
    names: tuple = FEATURES
    weights: tuple = (0.0,) * len(FEATURES)
    bias: float = 0.0
    threshold: float = 0.5           # best-F1 threshold on the training segment: NOT calibrated across segments
    top_fraction: float | None = None  # fraction of the training segment at or above ``threshold``
    params: dict = field(default_factory=dict)  # CrackleParams used at training
    info: dict = field(default_factory=dict)

    def logit(self, Z: np.ndarray) -> np.ndarray:
        w = np.asarray(self.weights, np.float32)
        return np.tensordot(w, Z, axes=1) + np.float32(self.bias)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore"):
            return (1.0 / (1.0 + np.exp(-self.logit(Z)))).astype(np.float32)

    def percentile_threshold(self, score: np.ndarray) -> float:
        """Per-segment operating point: the score quantile that flags the same fraction of this segment's
        scored pixels as ``threshold`` flagged on the training segment (falls back to ``threshold``)."""
        s = np.asarray(score)[np.isfinite(score)]
        if self.top_fraction is None or s.size == 0:
            return float(self.threshold)
        return float(np.quantile(s, 1.0 - self.top_fraction))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, txt: str) -> "CrackleModel":
        d = json.loads(txt)
        d["names"], d["weights"] = tuple(d["names"]), tuple(d["weights"])
        return cls(**d)

    @classmethod
    def fit(cls, Z: np.ndarray, label: np.ndarray, valid: np.ndarray, names=FEATURES, l2: float = 1e-3,
            n_samples: int = 400_000, seed: int = 0, params: CrackleParams | None = None,
            info: dict | None = None) -> "CrackleModel":
        """Newton/IRLS logistic regression on a random sample of valid pixels; the decision
        threshold is the one with the best pixel F1 on the training data itself."""
        ok = valid & np.all(np.isfinite(Z), 0)
        idx = np.flatnonzero(ok.ravel())
        rng = np.random.default_rng(seed)
        if idx.size > n_samples:
            idx = rng.choice(idx, n_samples, replace=False)
        X = Z.reshape(len(Z), -1)[:, idx].T.astype(np.float64)
        y = label.ravel()[idx].astype(np.float64)
        X1 = np.hstack([X, np.ones((len(X), 1))])
        w = np.zeros(X1.shape[1])
        reg = l2 * len(X) * np.r_[np.ones(X.shape[1]), 0.0]
        for _ in range(50):
            z = np.clip(X1 @ w, -30, 30)
            pr = 1 / (1 + np.exp(-z))
            g = X1.T @ (pr - y) + reg * w
            Hm = (X1 * (pr * (1 - pr))[:, None]).T @ X1 + np.diag(reg + 1e-9)
            dw = np.linalg.solve(Hm, g)
            w -= dw
            if np.abs(dw).max() < 1e-7:
                break
        m = cls(tuple(names), tuple(float(v) for v in w[:-1]), float(w[-1]),
                params=asdict(params) if params is not None else {}, info=dict(info or {}))
        sc = m.predict(Z)
        ev = binary_scores(sc[ok], label[ok])
        m.threshold = float(ev["best_f1_threshold"])
        m.top_fraction = float(np.mean(sc[ok] >= m.threshold))
        m.info.update({"train_auc": ev["auc"], "train_ap": ev["ap"], "train_best_f1": ev["best_f1"],
                       "n_samples": int(len(X))})
        return m


# Fitted by scripts/crackle_eval.py on segment 20230531121653 (official VC layers 16..48, surface 32) with the
# default CrackleParams, on pixels > 64 px from the mask edge; README "Ink-side fine-valley texture" has its
# cross-segment test numbers.
DEFAULT_MODEL_JSON = (
    "{\"names\":[\"crack_energy\",\"crack_isotropy\",\"coherence_loss\",\"relief\",\"smoothness\"],\"weights\":[1.2"
    "13603,-0.173073,-0.003471,-0.095651,0.038264],\"bias\":-2.494734,\"threshold\":0.217289,\"top_fraction\":0.143"
    "65,\"params\":{\"surface\":null,\"ink_side\":1,\"step\":1.0,\"band\":[3.0,7.0],\"ref_band\":[-6.0,-2.0],\"surf"
    "ace_band\":[-1.0,1.0],\"fiber_band\":[-1.0,1.0],\"sigma_crack\":1.5,\"sigma_pool\":16.0,\"sigma_bg\":192.0,\"f"
    "iber_width\":2.0,\"sigma_fine\":[0.7,2.0],\"edge_margin\":64.0},\"info\":{\"train_segment\":\"20230531121653\""
    ",\"train_auc\":0.793,\"train_ap\":0.3716,\"train_best_f1\":0.4257,\"n_samples\":400000,\"block_threshold\":0.1"
    "888,\"block_top_fraction\":0.1731}}"
)


def default_model() -> CrackleModel:
    if DEFAULT_MODEL_JSON is None:
        raise RuntimeError("no default crackle model bundled; fit one with CrackleModel.fit")
    return CrackleModel.from_json(DEFAULT_MODEL_JSON)


def crackle_score(stack, valid=None, model: CrackleModel | None = None, params: CrackleParams | None = None):
    """Texture score map (H, W) in [0, 1] and the feature dict.  NaN outside valid and within
    ``params.edge_margin`` px of its edge; ``feats["scored"]`` marks the scored pixels."""
    model = model or default_model()
    if params is None:
        params = CrackleParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in model.params.items()})
    feats = crackle_features(stack, valid, params, features=model.names)
    Z = standardize(feats, model.names)
    sc = model.predict(Z)
    scored = feats["valid"].copy()
    if params.edge_margin > 0:
        scored &= edge_distance(scored) > params.edge_margin
    sc[~scored] = np.nan
    feats["scored"] = scored
    return sc, feats


# --------------------------------------------------------------------------------------- io
def read_layers(path: str):
    """Load a layer stack from a directory of NN.tif files (VC / ``unroller render`` layout) or from a
    ``render_stream`` zarr.  Returns (stack (n, H, W), layer numbers): for tifs the numbers in the file
    names (so official layers 16..48 keep their numbers), for a zarr 0..n-1."""
    import glob
    import os

    import tifffile

    if os.path.exists(os.path.join(path, ".zarray")):
        import numcodecs

        with open(os.path.join(path, ".zarray")) as f:
            m = json.load(f)
        n, H, W = m["shape"]
        _, th, tw = m["chunks"]
        codec = numcodecs.get_codec(m["compressor"])
        out = np.zeros((n, H, W), np.dtype(m["dtype"]))
        for ty in range(-(-H // th)):
            for tx in range(-(-W // tw)):
                fp = os.path.join(path, "0", str(ty), str(tx))
                if os.path.exists(fp):
                    with open(fp, "rb") as f:
                        blk = np.frombuffer(codec.decode(f.read()), out.dtype).reshape(n, th, tw)
                    h, w = min(th, H - ty * th), min(tw, W - tx * tw)
                    out[:, ty * th:ty * th + h, tx * tw:tx * tw + w] = blk[:, :h, :w]
        return out, list(range(n))
    files = sorted(glob.glob(os.path.join(path, "*.tif")), key=lambda f: int(os.path.basename(f).split(".")[0]))
    if not files:
        raise FileNotFoundError(f"no NN.tif layers or zarr in {path}")
    nums = [int(os.path.basename(f).split(".")[0]) for f in files]
    if nums != list(range(nums[0], nums[0] + len(nums))):
        raise ValueError(f"layer numbers in {path} are not contiguous: {nums}")
    return np.stack([tifffile.imread(f) for f in files]), nums


# --------------------------------------------------------------------------------------- scoring
def binary_scores(score: np.ndarray, label: np.ndarray, threshold: float | None = None) -> dict:
    """ROC AUC (ties averaged), average precision, best F1 (and its threshold), F1 at ``threshold``.

    ``score`` and ``label`` are 1-D (already restricted to the evaluated pixels / blocks)."""
    s = np.asarray(score, np.float64).ravel()
    y = np.asarray(label, bool).ravel()
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    P, N = int(y.sum()), int((~y).sum())
    out = {"n": int(s.size), "positives": P, "prevalence": P / max(s.size, 1)}
    if P == 0 or N == 0:
        return out | {"auc": None, "ap": None, "best_f1": None, "best_f1_threshold": None}
    o = np.argsort(-s, kind="stable")
    s, y = s[o], y[o]
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]  # last index of each distinct score
    tp = np.cumsum(y)[last].astype(np.float64)
    fp = (last + 1 - tp)
    tpr, fpr = np.r_[0, tp / P], np.r_[0, fp / N]
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    prec = tp / (tp + fp)
    rec = tp / P
    ap = float(np.sum((rec - np.r_[0, rec[:-1]]) * prec))
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    out |= {"auc": auc, "ap": ap, "best_f1": float(f1[i]), "best_f1_threshold": float(s[last[i]])}
    if threshold is not None:
        pr = s >= threshold
        tp1 = int((pr & y).sum())
        out["f1_at_threshold"] = 2 * tp1 / max(int(pr.sum()) + P, 1)
        out["threshold"] = float(threshold)
    return out


def block_reduce(score: np.ndarray, label: np.ndarray, valid: np.ndarray, block: int = 32,
                 min_valid: float = 0.5, min_ink: float = 0.5):
    """Mean score and majority label per block x block tile (tiles >= min_valid inside ``valid``)."""
    H, W = score.shape
    h, w = H // block, W // block

    def tiles(a):
        return a[:h * block, :w * block].reshape(h, block, w, block).sum((1, 3), dtype=np.float64)

    v = valid & np.isfinite(score)
    nv = tiles(v)
    ssum = tiles(np.where(v, score, 0))
    lsum = tiles(v & label)
    keep = nv >= min_valid * block * block
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = ssum / nv
        bl = lsum / nv >= min_ink
    return bs[keep], bl[keep]


# --------------------------------------------------------------------------------------- overlay
def _colormap(t):
    """Perceptually ordered dark-red -> orange -> yellow ramp for t in [0, 1] (RGB float)."""
    stops = np.array([[0.0, 0.25, 0.0, 0.05], [0.35, 0.75, 0.05, 0.05], [0.7, 1.0, 0.55, 0.0],
                      [1.0, 1.0, 1.0, 0.45]])
    t = np.clip(t, 0, 1)
    return np.stack([np.interp(t, stops[:, 0], stops[:, c]) for c in (1, 2, 3)], -1)


def overlay(base: np.ndarray, score: np.ndarray, label=None, valid=None, path: str | None = None,
            vmin: float | None = None, vmax: float | None = None, alpha: float = 0.6,
            max_side: int | None = 4096, contour_rgb=(0, 230, 255)) -> np.ndarray:
    """RGB uint8 image: grey ``base`` (percentile-stretched) with ``score`` as a colour overlay whose
    opacity grows with the score above ``vmin`` (default: the median score), optional ``label``
    contours (cyan).  Downsampled by an integer factor so the longer side is <= max_side.
    Written to ``path`` (PNG) if given."""
    from PIL import Image

    base = np.asarray(base, np.float32)
    score = np.asarray(score, np.float32)
    valid = (np.ones(base.shape, bool) if valid is None else np.asarray(valid, bool)) \
        & np.isfinite(base) & np.isfinite(score)
    f = 1
    if max_side and max(base.shape) > max_side:
        f = int(np.ceil(max(base.shape) / max_side))
    if f > 1:  # area-average f x f cells (the ragged last row / column is dropped)
        H, W = (base.shape[0] // f) * f, (base.shape[1] // f) * f

        def ds(a):
            return a[:H, :W].reshape(H // f, f, W // f, f).mean((1, 3), dtype=np.float32)
        w = ds(valid.astype(np.float32))
        base = ds(np.where(valid, base, 0)) / np.maximum(w, 1e-6)
        score = ds(np.where(valid, score, 0)) / np.maximum(w, 1e-6)
        label = ds(np.asarray(label, np.float32)) > 0.5 if label is not None else None
        valid = w > 0.5
    b = base[valid]
    lo, hi = (np.percentile(b, [1, 99]) if b.size else (0.0, 1.0))
    g = np.clip((base - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = np.repeat(g[..., None], 3, -1)
    s = score[valid]
    if vmin is None:
        vmin = float(np.median(s)) if s.size else 0.0
    if vmax is None:
        vmax = float(np.percentile(s, 99.5)) if s.size else 1.0
    t = np.clip((np.nan_to_num(score, nan=vmin) - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    a = (alpha * t)[..., None] * valid[..., None]
    rgb = rgb * (1 - a) + _colormap(t) * a
    rgb[~valid] = 0.0
    if label is not None:
        L = np.asarray(label, bool)
        edge = L & ~binary_erosion(L, iterations=max(1, 2 // f + 1))
        rgb[edge] = np.asarray(contour_rgb, np.float32) / 255
    img = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    if path:
        Image.fromarray(img).save(path)
    return img
