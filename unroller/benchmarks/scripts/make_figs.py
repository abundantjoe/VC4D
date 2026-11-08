"""Make the README figures in docs/figs from public Scroll 1 data and the benchmark JSONs.
usage: python -I make_figs.py DATA RESULTS_DIR FIG_DIR
DATA holds seg_uv/ (obj2tifxyz --uv-size 2068 4102), official/{32,48}.tif (the published layers of segment
20230504093154) and zcache/ (chunk cache); RESULTS_DIR holds bench_render_512.json and bench_flatten_400.json.
libigl (pip install libigl==2.6.3) is optional: without it the distortion figure has two panels."""
import sys, os, json
# requires: pip install -e . matplotlib
import numpy as np, tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from unroller.surface import Surface
from unroller.volume import Volume
from unroller.render import render
from unroller.flatten import flatten, _clean, _grid_folds
from unroller.metrics import triangle_singular_values

D, RES, FIG = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(FIG, exist_ok=True)
URL = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, GRAY = "#2a78d6", "#a3a29c"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF})

def stretch(a, m=None):
    v = a[np.isfinite(a)] if m is None else a[m]
    lo, hi = np.percentile(v, [1, 99]); return np.clip((a - lo) / (hi - lo), 0, 1)

def ncc(a, b):
    m = np.isfinite(a) & (b > 0); a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

# 1) render comparison: official layers vs unroller, same 512 x 512 crop of the UV grid (as ncc_official.py)
X0, Y0, W, H = 384, 2024, 512, 512
s = Surface.load(f"{D}/seg_uv"); v = Volume(URL, disk_cache=D + "/zcache")
st = render(s, v, 3, step=16.0, crop=(X0, Y0, W, H), normal_sigma=6.0)
fig, ax = plt.subplots(2, 2, figsize=(7.2, 7.6))
for i, (L, k) in enumerate(((32, 1), (48, 2))):
    off = tifffile.imread(f"{D}/official/{L}.tif").astype(np.float64)[Y0:Y0 + H, X0:X0 + W]
    r = ncc(st[k], off)
    ax[i, 0].imshow(stretch(off, off > 0), cmap="gray", interpolation="nearest")
    ax[i, 0].set_title(f"official layer {L}", fontsize=10)
    ax[i, 1].imshow(stretch(st[k]), cmap="gray", interpolation="nearest")
    ax[i, 1].set_title(f"unroller layer {L}  (NCC {r:.3f})", fontsize=10)
for a in ax.flat: a.set_xticks([]); a.set_yticks([])
fig.suptitle("Scroll 1 segment 20230504093154, 512 x 512 crop, standardized 7.91 um zarr", fontsize=10, color=INK2)
fig.tight_layout(); fig.savefig(f"{FIG}/render_comparison.png", dpi=100); plt.close(fig)

# 2) flattening distortion maps: per-triangle log(area 2D / area 3D) on the 400 x 400 vertex crop
FX, FY, N = 420, 2080, 400
c = Surface(s.points[FY:FY + N, FX:FX + N])
idx, V, T = c.triangles(); T = _clean(V, T)
rr, cc = np.nonzero(idx >= 0); assert np.array_equal(idx[rr, cc], np.arange(len(rr)))
def logarea(uv):
    mir = _grid_folds(c, idx, V, T)
    sv, det = triangle_singular_values(V, T, uv, mir)
    la = np.log(sv[:, 0] * sv[:, 1]); flip = np.sign(det) != np.sign(np.median(det))
    acc = np.zeros(idx.shape); cnt = np.zeros(idx.shape)
    np.add.at(acc, (rr[T[:, 0]], cc[T[:, 0]]), la); np.add.at(cnt, (rr[T[:, 0]], cc[T[:, 0]]), 1)
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan), np.abs(la), int(flip.sum())
uv_vc = np.stack([cc, rr], 1).astype(np.float64)
a3 = 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
e1, e2 = uv_vc[T[:, 1]] - uv_vc[T[:, 0]], uv_vc[T[:, 2]] - uv_vc[T[:, 0]]
uv_vc *= np.sqrt(a3.sum() / (0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()))
panels = [("original VC UV map", uv_vc)]
g = flatten(c); panels.append(("unroller flatten (defaults)", g[rr, cc].astype(np.float64)))
try:
    import igl
    data = igl.ARAPData(); data.energy = igl.ARAP_ENERGY_TYPE_ELEMENTS; data.max_iter = 150
    F = np.ascontiguousarray(T, dtype=np.int64); b = np.array([0], np.int32)
    igl.arap_precomputation(np.ascontiguousarray(V, np.float64), F, 2, b, data)
    panels.append(("libigl ARAP, 150 iterations", igl.arap_solve(uv_vc[[0]], data, uv_vc.copy())))
except ImportError:
    pass
cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
fig, ax = plt.subplots(1, len(panels), figsize=(3.4 * len(panels) + 0.8, 3.9))
for a, (name, uv) in zip(np.atleast_1d(ax), panels):
    m, absla, nflip = logarea(uv)
    im = a.imshow(m, cmap=cmap, vmin=-0.2, vmax=0.2, interpolation="nearest")
    p50, p95 = np.percentile(absla, [50, 95])
    a.set_title(f"{name}\n|log area| p50 {p50:.4f}, p95 {p95:.4f}, {nflip} flips", fontsize=9)
    a.set_xticks([]); a.set_yticks([])
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cb.set_label("log(area 2D / area 3D)")
cb.outline.set_edgecolor(GRID)
fig.savefig(f"{FIG}/flatten_distortion.png", dpi=110, bbox_inches="tight", pad_inches=0.15); plt.close(fig)

# 3) timing bars from the benchmark JSONs (one panel per measure, each with its own axis)
R = json.load(open(f"{RES}/bench_render_512.json")); Fj = json.load(open(f"{RES}/bench_flatten_400.json"))
sets = [
    ("Render kernel throughput\n(M samples/s, 1 thread, higher is better)",
     [("unroller", R["unroller_inmem_t1_Mpts_s"]), ("torch grid_sample", R["torch_t1_Mpts_s"]),
      ("scipy map_coordinates", R["scipy_Mpts_s"])], "{:.1f}"),
    ("Crop render, 17.0 M samples\n(s, lower is better)",
     [("remote, cold", R["unroller_render_remote_cold_s"]), ("disk cache", R["unroller_render_diskcache_s"]),
      ("RAM cache", R["unroller_render_ramcache_s"])], "{:.1f} s"),
    ("Flatten 160k vertices to equal quality\n(s, lower is better)",
     [("unroller flatten", Fj["unroller_flatten_s"]), ("libigl ARAP, 150 it.", Fj["libigl"]["iter150"]["cum_s"])],
     "{:.1f} s"),
]
fig, ax = plt.subplots(1, 3, figsize=(12, 2.9))
for a, (title, rows, fmt) in zip(ax, sets):
    names = [r[0] for r in rows][::-1]; vals = [r[1] for r in rows][::-1]
    cols = [BLUE if (n.startswith("unroller") or title.startswith("Crop")) else GRAY for n in names]
    a.barh(names, vals, color=cols, height=0.55, edgecolor=SURF, linewidth=2)
    for y, val in enumerate(vals): a.text(val, y, "  " + fmt.format(val), va="center", color=INK, fontsize=9)
    a.set_xlim(0, max(vals) * 1.3); a.set_title(title, fontsize=9.5, loc="left")
    a.xaxis.grid(True, color=GRID, linewidth=0.8); a.set_axisbelow(True)
    for sp in ("top", "right", "left"): a.spines[sp].set_visible(False)
    a.tick_params(axis="y", length=0)
fig.text(0.01, 0.01, "Public Scroll 1, 4 shared CPU cores (load 7-9), nice 10, 2 numba threads", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(f"{FIG}/timing.png", dpi=110); plt.close(fig)
print("wrote", sorted(os.listdir(FIG)))
