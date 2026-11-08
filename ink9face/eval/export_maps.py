"""Convert the authors' cached two-face maps into the public maps layout read by reproduce.py (authors' side only).

    python eval/export_maps.py --scratch <cache tree> --v8 <v8-in result tree> --out <maps dir> [--new <dir from infer_two_faces.py>]

This is how the maps bundle was built.  It only renames and re-orients arrays (values are copied unchanged:
uint8 for the villa-loader models, float16 for v8-in), so a table computed from the bundle equals the table computed
from the cache.  Cached maps are in 'toward axis' order (cur = index increasing toward the scroll axis, rev = the
other order); a window's label meta 'flipped' says whether cur is the reversed native (+n) order.
"""
import argparse
import glob
import json
import os

import numpy as np

KEYS = dict(ink9_42="ink9_42", ink9_43="ink9_43", klavis="ink9_klavis_all7", reader="ink9_reader_v3", trace="ink9_trace_v3B")


def sources(S, V8):
    """window -> (label file, cur file, rev file, cur key suffix, rev key suffix, models, v8 (cur, rev) or None)."""
    J = {}
    fa = json.load(open(os.path.join(S, "face_all.json")))["rows"]
    for w in sorted(fa):
        J[w] = (f"{S}/heldout_v2/{w}.npz", f"{S}/heldout_maps/{w}.npz", f"{S}/rev/{w}.npz", "", "_s+0",
                ("ink9_42", "ink9_43", "klavis", "trace"), (f"{V8}/results/{w}.npz", f"{V8}/results_rev/{w}.npz"))
    for f in sorted(glob.glob(f"{S}/heldout0814_v2/*_w*.npz")):
        w = os.path.basename(f)[:-4]
        J[w] = (f, f"{S}/m0814/cur/{w}.npz", f"{S}/m0814/rev/{w}.npz", "_s+0", "_s+0",
                ("ink9_42", "ink9_43", "klavis", "reader", "trace"), (f"{V8}/results_0814_cur/{w}.npz", f"{V8}/results_0814_rev/{w}.npz"))
    for d in ("heldout0343P_v2", "heldout0009B_v2"):
        for f in sorted(glob.glob(f"{S}/{d}/*_w*.npz")):
            w = os.path.basename(f)[:-4]
            J[w] = (f, f"{S}/m2/cur/{w}.npz", f"{S}/m2/rev/{w}.npz", "_s+0", "_s+0", ("trace",), None)
    for f in sorted(glob.glob(f"{S}/set3/heldout0139_v2/*_w*.npz")):
        w = os.path.basename(f)[:-4]
        J[w] = (f, f"{S}/set3/face/cur/{w}.npz", f"{S}/set3/face/rev/{w}.npz", "_s+0", "_s+0", ("trace",), None)
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--v8", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "windows.json"))
    a = ap.parse_args()
    wanted = {x["w"] for x in json.load(open(a.windows))["windows"]}
    n = 0
    for w, (lab, cur, rev, sc, sr, mods, v8) in sources(a.scratch, a.v8).items():
        if w not in wanted:
            continue
        flipped = bool(json.loads(str(np.load(lab)["meta"])).get("old_meta", {}).get("flipped"))
        if os.path.exists(cur) and os.path.exists(rev):
            C, R = np.load(cur), np.load(rev)
            for mod in mods:
                kc, kr = KEYS[mod] + sc, KEYS[mod] + sr
                if kc in C.files and kr in R.files:
                    nat, anti = (R[kr], C[kc]) if flipped else (C[kc], R[kr])
                    os.makedirs(os.path.join(a.out, mod), exist_ok=True)
                    np.savez_compressed(os.path.join(a.out, mod, w + ".npz"), native=nat, anti_normal=anti); n += 1
        if v8 and all(os.path.exists(f) for f in v8):
            A, B = (np.load(f) for f in v8)
            nat, anti = (B, A) if flipped else (A, B)
            os.makedirs(os.path.join(a.out, "v8"), exist_ok=True)
            np.savez_compressed(os.path.join(a.out, "v8", w + ".npz"), native=nat["p"], anti_normal=anti["p"],
                                covered=A["covered"] & B["covered"]); n += 1
    print("wrote", n, "map pairs to", a.out)


if __name__ == "__main__":
    main()
