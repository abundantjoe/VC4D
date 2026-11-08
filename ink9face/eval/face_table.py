"""Face table: per-window two-face scores and the summary (rule agreement, paired AUC gain, scroll band vote).

Shared by eval_face.py (our cached tree), reproduce.py (public downloads + a maps folder) and the tests.

Public maps layout (what reproduce.py reads, what infer_two_faces.py and export_maps.py write):
    <maps>/<model>/<window>.npz   keys  native, anti_normal   (uint8 0..255 or float 0..1, H x W of the window)
                                        covered (optional bool: pixels the model actually predicted; v8-in only)
    native      = model fed the layer stack with index increasing along +n (VC3D normal; vc_render_tifxyz order)
    anti_normal = the same stack reversed
Model ids: ink9_42, ink9_43 (averaged into 'ink9avg'), klavis, reader (= Reader v3), v8, trace, and the newest
checkpoints reader_v2, reader_v3, d9v4C, dense_native.
"""
import json
import os

import numpy as np
from scipy.stats import binomtest

from ink9face.metrics import letter_band_energy, mesh_paired, score

MODELS = ("ink9avg", "klavis", "reader", "v8", "trace")       # the headline table (results/face_eval.json)
CLEAN = ("ink9avg", "klavis", "reader", "v8")                 # models that may define the clean-model truth
NEW_MODELS = ("reader_v2", "reader_v3", "d9v4C", "dense_native")
PARTS = {"ink9avg": ("ink9_42", "ink9_43")}
EXCL_KEY = {"ink9avg": ("ink9_42", "ink9_43"), "reader": ("reader_v3",)}
BAND = ("klavis", "ink9avg", "trace")
METRICS = ("auc", "auc_37um", "letter")


def U(a):
    a = np.asarray(a).astype(np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def load_exclusions(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "exclusions.json")
    return json.load(open(path))["trained_on"]


def trained(excl, scroll, model, seg):
    """True if `model` saw segment `seg` of `scroll` in training (prefix match on public training lists)."""
    t = excl.get(scroll, {})
    for k in EXCL_KEY.get(model, (model,)):
        v = t.get(k, [])
        if v == "ALL" or any(seg.startswith(p) or p.startswith(seg) for p in v):
            return True
    return False


def load_maps(maps_dir, w, models):
    """{model: (p_native, p_anti, covered|None)} for every model with both faces of window w."""
    out = {}
    for mod in models:
        parts = PARTS.get(mod, (mod,))
        fs = [os.path.join(maps_dir, p, w + ".npz") for p in parts]
        if not all(os.path.exists(f) for f in fs):
            continue
        Z = [np.load(f) for f in fs]
        if not all("native" in z.files and "anti_normal" in z.files for z in Z):
            continue
        pn = np.mean([U(z["native"]) for z in Z], 0); pa = np.mean([U(z["anti_normal"]) for z in Z], 0)
        cov = Z[0]["covered"] if "covered" in Z[0].files else None
        out[mod] = (pn, pa, cov)
    return out


def score_window(scroll, seg, ink, mask, toward_axis_is_reversed, maps, contaminated, models=MODELS, clean=CLEAN):
    """One row of the face table.  maps = {model: (p_native, p_anti_normal, covered|None)}."""
    y = ink.astype(bool); m = mask.astype(bool)
    r = dict(scroll=scroll, seg=seg, flipped=bool(toward_axis_is_reversed), contaminated=list(contaminated), scores={}, band={})
    for mod in models:
        if mod not in maps:
            continue
        pn, pa, cov = maps[mod]
        mm = m & cov if cov is not None else m
        sn, sa = score(pn, mm, y), score(pa, mm, y)
        r["scores"][mod] = dict(native=sn, reversed=sa, toward_axis=sa if toward_axis_is_reversed else sn)
        if mod in BAND:
            r["band"][mod] = [letter_band_energy(pn, m), letter_band_energy(pa, m)]
    cl = [k for k in r["scores"] if k in clean and k not in r["contaminated"]]
    if cl:
        g = np.mean([r["scores"][k]["reversed"]["auc"] - r["scores"][k]["native"]["auc"] for k in cl])
        r["truth"] = dict(source="clean:" + "+".join(cl), reversed=bool(g > 0), gap=float(g))
    if "trace" in r["scores"]:
        g = r["scores"]["trace"]["reversed"]["auc"] - r["scores"]["trace"]["native"]["auc"]
        r["truth_trace"] = dict(reversed=bool(g > 0), gap=float(g))
    return r


def summarise(rows, models=MODELS):
    out = {"agreement": {}, "models": {}}
    scrolls = sorted({r["scroll"] for r in rows.values()})
    # 1. how often is each label-free rule right?
    for truth_key in ("truth", "truth_trace"):
        for sc in scrolls + ["ALL"]:
            ids = [w for w, r in rows.items() if truth_key in r and (sc == "ALL" or r["scroll"] == sc)]
            if not ids:
                continue
            T = np.array([rows[w][truth_key]["reversed"] for w in ids])
            rules = dict(anti_normal=np.ones(len(ids), bool), native=np.zeros(len(ids), bool),
                         toward_axis=np.array([rows[w]["flipped"] for w in ids]))
            if truth_key == "truth" and all("klavis" in rows[w]["band"] for w in ids):
                rules["band_klavis"] = np.array([rows[w]["band"]["klavis"][1] > rows[w]["band"]["klavis"][0] for w in ids])
            d = {}
            for nm, pick in rules.items():
                k = int((pick == T).sum())
                d[nm] = dict(correct=k, n=len(ids), p=float(binomtest(k, len(ids), 0.5, alternative="greater").pvalue))
            d["wrong_anti_normal"] = [dict(w=w, gap=rows[w][truth_key]["gap"]) for w in ids if not rows[w][truth_key]["reversed"]]
            out["agreement"][f"{truth_key}:{sc}"] = d
    # 2. AUC / letter under each choice, paired anti_normal - native and anti_normal - toward_axis (mesh bootstrap)
    for mod in models:
        for sc in scrolls + ["ALL"]:
            ids = [w for w, r in rows.items() if mod in r["scores"] and mod not in r["contaminated"] and (sc == "ALL" or r["scroll"] == sc)]
            if not ids:
                continue
            meshes = [rows[w]["seg"] for w in ids]
            res = dict(n_windows=len(ids), n_meshes=len(set(meshes)))
            for met in METRICS:
                S = {w: rows[w]["scores"][mod] for w in ids}
                vals = {c: np.array([S[w][c][met] for w in ids]) for c in ("native", "reversed", "toward_axis")}
                orc = np.array([S[w]["reversed" if S[w]["reversed"]["auc"] > S[w]["native"]["auc"] else "native"][met] for w in ids])
                if met == "auc":
                    orc = np.maximum(vals["native"], vals["reversed"])
                res[met] = dict(native=float(np.nanmean(vals["native"])), anti_normal=float(np.nanmean(vals["reversed"])),
                                toward_axis=float(np.nanmean(vals["toward_axis"])), self_oracle=float(np.nanmean(orc)),
                                anti_minus_native=mesh_paired(vals["reversed"] - vals["native"], meshes),
                                anti_minus_toward_axis=mesh_paired(vals["reversed"] - vals["toward_axis"], meshes))
            out["models"][f"{mod}:{sc}"] = res
    # 3. band rule at scroll level (vote over windows)
    out["band_scroll_vote"] = {}
    for sc in scrolls:
        for bm in BAND:
            ids = [w for w, r in rows.items() if r["scroll"] == sc and bm in r["band"]]
            if ids:
                s = float(np.sum([np.log(rows[w]["band"][bm][1] / rows[w]["band"][bm][0]) for w in ids]))
                out["band_scroll_vote"][f"{sc}:{bm}"] = dict(n=len(ids), log_ratio_sum=s, says="reversed" if s > 0 else "native",
                                                            windows_saying_reversed=int(sum(rows[w]["band"][bm][1] > rows[w]["band"][bm][0] for w in ids)))
    return out


def print_summary(S):
    for k, v in S["agreement"].items():
        print(k, {n: f"{x['correct']}/{x['n']}" for n, x in v.items() if isinstance(x, dict)})
    for k, v in S["models"].items():
        a_ = v["auc"]; l_ = v["letter"]; d = a_["anti_minus_native"]
        print(f"{k:24s} {v['n_windows']:3d}w {v['n_meshes']:3d}m AUC native {a_['native']:.3f} anti {a_['anti_normal']:.3f} "
              f"axis {a_['toward_axis']:.3f} d(anti-native) {d['mean']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] "
              f"{d['better']}/{d['n_meshes']} | letter {l_['native']:+.3f} -> {l_['anti_normal']:+.3f}")


def compare(a, b, tol=1e-6, path=""):
    """List of differences between two summary JSON trees (numbers compared with abs tolerance)."""
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append(f"{path}/{k}: missing in {'new' if k not in a else 'reference'}")
            else:
                diffs += compare(a[k], b[k], tol, f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs += compare(x, y, tol, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (np.isnan(a) and np.isnan(b)) and abs(a - b) > tol:
            diffs.append(f"{path}: {a} != {b}")
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs
