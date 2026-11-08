import sys, os, subprocess, numpy as np, tifffile
R = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/20230531121653"
OUT = sys.argv[1]; d = OUT; out = []
for L in range(17, 43):
    f = f"{d}/{L:02d}.tif"
    subprocess.run(["curl", "-sSfo", f, f"{R}/layers/{L:02d}.tif"], check=True)
    a = tifffile.imread(f); out.append((a >> 8).astype(np.uint8)); os.remove(f)  # cv2.imread(...,0) on uint16 == >>8
    print(L, a.shape, a.dtype, flush=True)
np.save(OUT + "/official_17_42_u8.npy", np.stack(out))
subprocess.run(["curl", "-sSfo", OUT + "/mask.png", f"{R}/20230531121653_mask.png"], check=True)
