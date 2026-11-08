import sys, json; pass  # requires: pip install -e .
import numpy as np, tifffile
from unroller.surface import Surface
from unroller.volume import Volume
from unroller.render import render
D, seg, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
X0, Y0, W, H = 384, 2024, 512, 512
URL = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr"
s = Surface.load(f"{D}/{seg}"); v = Volume(URL, disk_cache=D + "/zcache")
off = {16: 0, 32: 1, 48: 2}
offi = {l: tifffile.imread(f"{D}/official/{l}.tif").astype(np.float64) for l in off}
def ncc(a, b):
    m = np.isfinite(a) & (b > 0); a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
R = {"segment": seg}
for sig in (6.0, 0.0):
    st = render(s, v, 3, step=16.0, crop=(X0, Y0, W, H), normal_sigma=sig)
    if sig == 6.0:  # register on layer 32 (offset 0, normal-independent)
        best = max(((ncc(st[1], offi[32][Y0 + dy:Y0 + dy + H, X0 + dx:X0 + dx + W]), dy, dx)
                    for dy in range(-40, 41, 1) for dx in range(-40, 41, 1)))
        _, dy, dx = best; R["shift_dy_dx"] = [dy, dx]
    for l, i in off.items():
        R[f"ncc_L{l}_sigma{sig:g}"] = ncc(st[i], offi[l][Y0 + dy:Y0 + dy + H, X0 + dx:X0 + dx + W])
json.dump(R, open(OUT, "w"), indent=1); print(R)
