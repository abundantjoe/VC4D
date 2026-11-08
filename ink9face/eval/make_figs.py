"""Figures from the PUBLIC evaluation results (PHerc0500P2, PHerc0814, PHerc0343P, PHerc0009B, PHerc0139 only).

    python eval/make_figs.py --results results --scratch <dir> --ink <ink dir> --figs figs
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True); ap.add_argument("--scratch", required=True)
ap.add_argument("--ink", required=True); ap.add_argument("--figs", required=True)
a = ap.parse_args()
os.makedirs(a.figs, exist_ok=True)
# validated default categorical palette (dataviz skill reference instance), fixed order
C = dict(PHerc0500P2="#2a78d6", PHerc0814="#eb6834", PHerc0343P="#1baf7a", PHerc0009B="#eda100", PHerc0139="#e87ba4")
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
F = json.load(open(os.path.join(a.results, "face_eval.json")))
rows, S = F["rows"], F["summary"]

# ---- Fig 1: per-window AUC, native (+n) vs anti-normal, per model
panels = [("ink9avg", "organizers' ink_9um (seeds 42+43)"), ("klavis", "KLAVIS"), ("v8", "v8-in"), ("reader", "Reader v3 (clean windows)"),
          ("trace", "TRACE (in-sample face probe only)")]
fig, ax = plt.subplots(1, 5, figsize=(16, 4.0))
for k, (mod, title) in enumerate(panels):
    A = ax[k]
    seen = set()
    for w, r in rows.items():
        if mod in r["scores"] and mod not in r["contaminated"]:
            x, y = r["scores"][mod]["native"]["auc"], r["scores"][mod]["reversed"]["auc"]
            lab = r["scroll"] if r["scroll"] not in seen else None; seen.add(r["scroll"])
            A.scatter(x, y, s=22, color=C[r["scroll"]], edgecolor=SURF, linewidth=1.0, label=lab, zorder=3)
    A.plot([0.3, 1], [0.3, 1], color=MUTED, lw=1, ls="--", zorder=2)
    A.set_xlim(0.3, 1); A.set_ylim(0.3, 1); A.set_aspect("equal")
    A.set_title(title, fontsize=9, color=INK); A.set_xlabel("AUC, native order (+n)")
    if k == 0:
        A.set_ylabel("AUC, anti-normal order (-n)")
    s = S["models"].get(f"{mod}:ALL")
    if s:
        d = s["auc"]["anti_minus_native"]
        A.text(0.32, 0.97, f"d = {d['mean']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]\n{d['better']}/{d['n_meshes']} meshes",
               va="top", fontsize=8, color=INK)
    A.legend(loc="lower right", fontsize=7, frameon=False)
fig.suptitle("Same model, same window, layer order flipped: points above the diagonal = anti-normal order reads the inked face", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0.02, 1, 0.95)); fig.savefig(os.path.join(a.figs, "fig1_face_auc_per_window.png"), dpi=150); plt.close(fig)

# ---- Fig 2: rule accuracy
keys = [("truth:PHerc0500P2", "PHerc0500P2\nclean-model truth"), ("truth:PHerc0814", "PHerc0814\nclean-model truth"),
        ("truth_trace:PHerc0343P", "PHerc0343P\nTRACE oracle"), ("truth_trace:PHerc0009B", "PHerc0009B\nTRACE oracle"),
        ("truth_trace:PHerc0139", "PHerc0139\nTRACE oracle")]
rules = [("anti_normal", "anti-normal (-n)", "#2a78d6"), ("native", "native (+n)", "#eb6834"), ("toward_axis", "toward axis", "#1baf7a"),
         ("band_klavis", "two-face band (KLAVIS)", "#eda100")]
fig, A = plt.subplots(figsize=(10, 3.6))
wbar = 0.2
for i, (rk, rl, col) in enumerate(rules):
    xs, ys = [], []
    for j, (kk, _) in enumerate(keys):
        d = S["agreement"].get(kk, {}).get(rk)
        if d:
            xs.append(j + (i - 1.5) * wbar); ys.append(d["correct"] / d["n"])
            A.text(xs[-1], ys[-1] + 0.01, f"{d['correct']}/{d['n']}", ha="center", fontsize=7, color=MUTED)
    A.bar(xs, ys, width=wbar - 0.03, color=col, label=rl, zorder=3)
A.axhline(0.5, color=MUTED, lw=1, ls="--")
A.set_xticks(range(len(keys))); A.set_xticklabels([k[1] for k in keys], fontsize=8)
A.set_ylim(0, 1.08); A.set_ylabel("share of windows where the rule picks the inked face")
A.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
fig.tight_layout(); fig.savefig(os.path.join(a.figs, "fig2_face_rule_accuracy.png"), dpi=150); plt.close(fig)

# ---- Fig 3: example window (PHerc0500P2, public): CT, label, KLAVIS native vs anti-normal
w = max((w for w, r in rows.items() if r["scroll"] == "PHerc0500P2" and "klavis" in r["scores"]),
        key=lambda w: rows[w]["scores"]["klavis"]["reversed"]["auc"] - rows[w]["scores"]["klavis"]["native"]["auc"])
St = np.load(os.path.join(a.scratch, "heldout", w + ".npz")); L = np.load(os.path.join(a.scratch, "heldout_v2", w + ".npz"))
cur = np.load(os.path.join(a.scratch, "heldout_maps", w + ".npz"))["ink9_klavis_all7"].astype(np.float32)
rev = np.load(os.path.join(a.scratch, "rev", w + ".npz"))["ink9_klavis_all7_s+0"].astype(np.float32) / 255
fl = rows[w]["flipped"]; nat, anti = (rev, cur) if fl else (cur, rev)
m = L["valid"] & L["acc"]
sk = rows[w]["scores"]["klavis"]
fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
ims = [(St["st"][12].astype(np.float32), "CT, layer 0 (surface)", "gray"), (np.where(m, L["ink"], np.nan), "reference label, exact transfer (ink = black)", "gray_r"),
       (np.where(m, nat, np.nan), f"KLAVIS, native +n   AUC {sk['native']['auc']:.2f}", "magma"),
       (np.where(m, anti, np.nan), f"KLAVIS, anti-normal   AUC {sk['reversed']['auc']:.2f}", "magma")]
for A, (im, t, cm) in zip(ax, ims):
    A.imshow(im, cmap=cm); A.set_title(t, fontsize=9, color=INK); A.axis("off")
fig.suptitle(f"PHerc0500P2 {w} (public; 1 px = 9.362 um; window 9.9 mm)", fontsize=9, color=MUTED)
fig.tight_layout(); fig.savefig(os.path.join(a.figs, "fig3_example_window_PHerc0500P2.png"), dpi=130); plt.close(fig)

# ---- Fig 4: depth sweep (PHerc0500P2, correct face), from the earlier sweep results
sw = json.load(open(os.path.join(a.scratch, "eval_shift.json")))["summary"]
shifts = list(range(-4, 5))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for mod, nm, col in (("ink9_42", "ink_9um seed 42 (29 w)", "#2a78d6"), ("ink9_klavis_all7", "KLAVIS (29 w)", "#eb6834"), ("ink9_reader_v3", "Reader v3 (15 clean w)", "#1baf7a")):
    if mod in sw:
        ax[0].plot(shifts, sw[mod]["auc"], color=col, lw=2, marker="o", ms=5, label=nm)
        ax[1].plot(shifts, sw[mod]["letter"], color=col, lw=2, marker="o", ms=5, label=nm)
for A, yl in zip(ax, ("pixel AUC", "letter-scale r (minus shift null)")):
    A.axvline(0, color=MUTED, lw=1, ls="--"); A.set_xlabel("depth shift of the input window (voxels, + = deeper along the read direction)"); A.set_ylabel(yl)
ax[0].legend(fontsize=8, frameon=False)
fig.suptitle("PHerc0500P2, correct face: accuracy is flat for shifts 0..+3 and drops for negative shifts", fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(a.figs, "fig4_depth_sweep_PHerc0500P2.png"), dpi=150); plt.close(fig)

# ---- Fig 5: label-transfer residuals
Lb = json.load(open(os.path.join(a.results, "labels_eval.json")))["registration"]
fig, A = plt.subplots(figsize=(8, 3.2))
for j, (sc, d) in enumerate(Lb.items()):
    y = [r["node_p95"] for r in d["rows"]]
    A.scatter(j + np.random.default_rng(j).uniform(-0.15, 0.15, len(y)), y, s=16, color=C[sc], edgecolor=SURF, linewidth=1, zorder=3)
    A.text(j, max(y) + 0.08, f"{d['passed']}/{d['windows']} pass", ha="center", fontsize=8, color=MUTED)
A.axhline(2.0, color=MUTED, lw=1, ls="--"); A.text(len(Lb) - 0.5, 2.03, "pass threshold (p95 < 2 vx)", ha="right", fontsize=8, color=MUTED)
A.set_xticks(range(len(Lb))); A.set_xticklabels(list(Lb), fontsize=8); A.set_ylim(0, 2.3)
A.set_ylabel("3D node residual p95 (model-scan voxels)")
A.set_title("Geometry-only label transfer: |T P_coarse - P_label(rule)| per window", fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(a.figs, "fig5_label_transfer_residuals.png"), dpi=150); plt.close(fig)
print("figures written to", a.figs)
