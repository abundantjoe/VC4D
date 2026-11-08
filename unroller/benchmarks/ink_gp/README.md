# Ink end-use check: 2023 Grand Prize TimeSformer on unroller renders

Scores unroller layer stacks and the official layers with the same ink model and compares them against the public
Grand Prize hand labels. Results: `../results/ink_gp/`. Figures: `../../docs/figs/ink_gp_*.png`.

- **Model:** `scrollprize/timesformer_GP_scroll1` (MIT), loaded into `timesformer_pytorch` 0.4.1. Set `GP_MODEL` to
  its `model.safetensors`.
- **Preprocessing** follows `younader/Vesuvius-Grandprize-Winner`: layers 17-42 of a 65-layer stack (unroller
  `-n 31` layers 0-25 are the same offsets -15..+10), uint16 `>> 8`, clip 0-200, `/255`, 64 px tiles, 4x4 output
  upsampled x16 and Gaussian-blended. One change: bf16 autocast on CPU (max sigmoid difference to fp32 0.001 on
  random input).
- **Segments:** `20230531121653` (public GP labels, which were in the GP training set, so the absolute AUC is
  in-sample; the paired difference between renders is the quantity of interest) and `20230504093154` (no labels;
  correlation only).
- **Statistics:** paired block bootstrap over 256 px blocks, 200 resamples (`boot.py`).

```bash
pip install timesformer-pytorch==0.4.1 safetensors opencv-python-headless
SEG=20230531121653; ROOT=https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg
mkdir -p w && python get_official.py w                 # official layers 17-42 as uint8 + mask.png
curl -sSfo w/seg.obj $ROOT/paths/$SEG/$SEG.obj && unroller obj2tifxyz w/seg.obj w/seg_uv --uv-size <size of the official layer TIFFs>
unroller render w/seg_uv - w/unr16 -n 31 --no-tif --volume-16bit $ROOT/volumes/20230205180739
# convert w/unr16/layers.zarr[0:26] >> 8 to w/unr16_17_42_u8.npy (uint8, padded to the mask shape), then:
MASK=w/mask.png python run_pred.py w/official_17_42_u8.npy w/pred_official_s64.npy 64
MASK=w/mask.png python run_pred.py w/unr16_17_42_u8.npy    w/pred_unr16_s64.npy    64
python boot.py w <GP repo>/all_labels/${SEG}_inklabels.png bootstrap.json
```

`remap.py` maps uint8 layers onto the official histogram (the "remapped" row); it cannot restore values that the
standardized zarr clipped to 0.
