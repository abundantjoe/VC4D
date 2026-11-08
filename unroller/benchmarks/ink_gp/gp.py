"""GP 2023 TimeSformer (scrollprize/timesformer_GP_scroll1) inference, CPU, faithful to
younader/Vesuvius-Grandprize-Winner inference_timesformer.py: 26 layers, uint8 (cv2 16->8 = >>8),
clip 0..200, /255, 64 px tiles, stride 21, 4x4 output upsampled x16 bilinear, gaussian-weighted blend."""
import sys, time, numpy as np, torch, torch.nn.functional as F
import os
MODEL = os.environ.get("GP_MODEL", "dl_tsf/model.safetensors")  # scrollprize/timesformer_GP_scroll1 (MIT)
from timesformer_pytorch import TimeSformer
from safetensors.torch import load_file
import scipy.stats as st

def load_model():
    m = TimeSformer(dim=512, image_size=64, patch_size=16, num_frames=26, num_classes=16, channels=1,
                    depth=8, heads=6, dim_head=64, attn_dropout=0.1, ff_dropout=0.1)
    sd = {k[len("backbone."):]: v for k, v in load_file(MODEL).items()}
    missing, unexp = m.load_state_dict(sd, strict=False)
    assert not unexp, unexp
    assert all("rot_emb" in k for k in missing), missing
    return m.eval()

def gkern(k=64, nsig=1):
    x = np.linspace(-nsig, nsig, k + 1); kern1d = np.diff(st.norm.cdf(x))
    kern = np.outer(kern1d, kern1d); return kern / kern.max()

def predict(model, vol_u8, mask=None, stride=21, bs=32, log=None, bf16=True):
    """vol_u8: (26, H, W) uint8 already in GP convention (before clip). Returns (H, W) float in [0,1]."""
    _, H, W = vol_u8.shape
    v = np.clip(vol_u8, 0, 200).astype(np.float32) / 255.0
    ys = list(range(0, H - 64 + 1, stride)); xs = list(range(0, W - 64 + 1, stride))
    tiles = [(y, x) for y in ys for x in xs if mask is None or mask[y:y+64, x:x+64].all()]
    pred = np.zeros((H, W), np.float64); cnt = np.zeros((H, W), np.float64)
    ker = torch.tensor(gkern(64, 1), dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(tiles), bs):
            b = tiles[i:i+bs]
            x = torch.from_numpy(np.stack([v[:, y:y+64, xx:xx+64] for y, xx in b]))[:, None]  # B,1,26,64,64
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                out = model(x.permute(0, 2, 1, 3, 4))
            out = out.float().view(-1, 1, 4, 4)
            p = (F.interpolate(torch.sigmoid(out), scale_factor=16, mode="bilinear")[:, 0] * ker).numpy()
            for j, (y, xx) in enumerate(b):
                pred[y:y+64, xx:xx+64] += p[j]; cnt[y:y+64, xx:xx+64] += 1
            if log and (i // bs) % 20 == 0:
                print(f"{i+len(b)}/{len(tiles)} tiles {time.time()-t0:.0f}s", file=log, flush=True)
    pred /= np.clip(cnt, 1, None)
    return np.clip(np.nan_to_num(pred), 0, 1).astype(np.float32), cnt > 0
