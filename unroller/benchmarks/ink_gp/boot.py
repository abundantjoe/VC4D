import numpy as np, cv2, json
from scipy.stats import rankdata
import sys
# argv: folder with mask.png + pred_{official,unr16,unr_u8,unr_u8remap}_s64.npy; GP label PNG
# (younader/Vesuvius-Grandprize-Winner all_labels/20230531121653_inklabels.png); output JSON
S2, LAB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; lab = cv2.imread(LAB, 0)
m = cv2.imread(S2 + "/mask.png", 0) > 0; H, W = m.shape; lab = lab[:H, :W] > 127
P = {k: np.load(f"{S2}/pred_{k}_s64.npy") for k in ["official", "unr16", "unr_u8", "unr_u8remap"]}
v = m & np.all([np.isfinite(p) for p in P.values()], 0)
B = 256; by, bx = np.nonzero(v); blk = (by // B) * 100 + bx // B; ub = np.unique(blk)
y = lab[v]; vals = {k: p[v] for k, p in P.items()}
def auc(p, yy):
    r = rankdata(p); n1 = yy.sum(); n0 = len(yy) - n1; return (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
idx = {b: np.nonzero(blk == b)[0] for b in ub}; rng = np.random.default_rng(0); out = {k: [] for k in P}
for _ in range(200):
    s = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
    if y[s].sum() == 0: continue
    for k in P: out[k].append(auc(vals[k][s], y[s]))
res = {"n_blocks_256px": int(len(ub)), "n_boot": len(out["official"])}
for k in ["unr16", "unr_u8", "unr_u8remap"]:
    d = np.array(out[k]) - np.array(out["official"]); res[f"dAUC_{k}_minus_official"] = [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
print(json.dumps(res, indent=1)); json.dump(res, open(OUT, "w"), indent=1)
