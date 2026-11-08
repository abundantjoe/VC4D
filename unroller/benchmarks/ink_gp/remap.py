"""Monotone quantile remap of unroller uint8 layers onto the official layers' histogram (inside the mask).
Restores contrast/offset; cannot restore values clipped to 0 in the standardized zarr (they map to one level)."""
import sys, numpy as np, cv2
u = np.load(sys.argv[1]); o = np.load(sys.argv[2], mmap_mode="r"); m = cv2.imread(sys.argv[3], 0) > 0
hu = np.bincount(u[:, m].ravel(), minlength=256).astype(float); ho = np.bincount(np.asarray(o[:, m]).ravel(), minlength=256).astype(float)
cu = np.cumsum(hu) / hu.sum(); co = np.cumsum(ho) / ho.sum()
mid = np.concatenate([[0], cu[:-1]]) * 0.5 + cu * 0.5  # mid-rank of each unroller level
lut = np.clip(np.rint(np.interp(mid, co, np.arange(256))), 0, 255).astype(np.uint8)
np.save(sys.argv[4], lut[u]); print("lut", lut[[0, 1, 50, 100, 150, 200, 255]].tolist())
