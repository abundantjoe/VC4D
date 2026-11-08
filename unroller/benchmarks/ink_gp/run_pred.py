"""MASK=segment_mask.png python run_pred.py in.npy(26,H,W uint8) out.npy stride [y0 x0 h w] ; mask = official mask.png crop"""
import sys, numpy as np, torch, cv2, time
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0]); import gp
torch.set_num_threads(3)
v = np.load(sys.argv[1], mmap_mode="r"); stride = int(sys.argv[3])
import os; m = cv2.imread(os.environ["MASK"], 0) > 0
if len(sys.argv) > 4:
    y0, x0, h, w = map(int, sys.argv[4:8]); m = m[y0:y0+h, x0:x0+w]
    if v.shape[1:] != (h, w): v = v[:, y0:y0+h, x0:x0+w]
v = np.ascontiguousarray(v); assert m.shape == v.shape[1:], (m.shape, v.shape)
t = time.time(); p, cov = gp.predict(gp.load_model(), v, mask=m, stride=stride, log=sys.stderr)
np.save(sys.argv[2], np.where(cov, p, np.nan).astype(np.float32)); print(f"done {time.time()-t:.0f}s", file=sys.stderr)
