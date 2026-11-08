"""Negative control: can a CT-only statistic (no model) tell the inked face?  Uses the rendered public stacks of
PHerc0500P2 (29) and PHerc0139 (11), truth = eval_face.py (clean-model truth, else TRACE oracle).

    python eval/eval_ct_cue.py --scratch <dir> --face results/face_eval.json --out results/ct_cue.json
Cues (native +n frame): density asymmetry  mean z(layers +1..+6) - mean z(-6..-1);  density centroid offset;
texture asymmetry (per-layer std).  Reported: mean of each cue when anti-normal vs native is right, and the share of positive signs;
with only 2 'native' windows this is mostly a check that no cue explains those exceptions.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ink9face.depth import profile  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--scratch"); ap.add_argument("--face"); ap.add_argument("--out"); a = ap.parse_args()
rows = json.load(open(a.face))["rows"]; out = []
for w, r in rows.items():
    d = {"PHerc0500P2": "heldout", "PHerc0139": "set3/heldout0139"}.get(r["scroll"])
    f = os.path.join(a.scratch, d or "", w + ".npz")
    if not d or not os.path.exists(f):
        continue
    z = np.load(f); st = z["st"].astype(np.float32)[::-1] if r["flipped"] else z["st"].astype(np.float32); v = z["valid"]
    t = r.get("truth", r.get("truth_trace"))["reversed"]
    p = profile(st, v); c = np.arange(st.shape[0]) - st.shape[0] // 2; q = np.clip(p, 0, None)
    sd = np.array([st[i][v].std() for i in range(st.shape[0])])
    out.append(dict(w=w, scroll=r["scroll"], anti_right=bool(t), asym=float(p[13:19].mean() - p[6:12].mean()),
                    centroid=float((c * q).sum() / q.sum()), texture=float(sd[13:19].mean() - sd[6:12].mean())))
y = np.array([o["anti_right"] for o in out]); res = dict(n=len(out), anti_right=int(y.sum()))
for k in ("asym", "centroid", "texture"):
    x = np.array([o[k] for o in out])
    res[k] = dict(mean_when_anti_right=float(x[y].mean()), mean_when_native_right=float(x[~y].mean()) if (~y).any() else None,
                  sign_positive_share=float((x > 0).mean()))
json.dump(dict(summary=res, rows=out), open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))
