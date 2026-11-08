#!/usr/bin/env bash
# Usage example on one PUBLIC segment of PHerc0500P2 (Vesuvius Challenge open data; labels by the organizers).
# Downloads ~250 MB (label mesh, full-resolution label tif, coarse 9.362 um mesh, transform) into ./ink9face_cache.
set -euo pipefail
SCROLL=PHerc0500P2
SEG=20250611132918-z_dbg_gen_00302_inp_hr

# 1) exact 9.362 um labels for a 53 x 53-cell window (1061 x 1061 px at 1 px = 1 voxel), from the bucket
python -m ink9face labels --scroll $SCROLL --segment $SEG --cache ink9face_cache --crop 152 10 53 53 --out out/${SEG}_w0
#   -> out/<seg>_w0_ink.tif (uint8 label, 255 = ink), out/<seg>_w0_mask.tif (labelled & mesh valid), out/<seg>_w0.json
#      (transform used, LSQ cross-check, node residuals; exits 3 if the window fails the 1 / 2 voxel gate)

# 2) which layer order to feed a 9 um model for this mesh (VC3D normal convention), compared with the toward-axis rule
python -m ink9face face --mesh ink9face_cache/$SEG/coarse --crop 152 10 53 53 --axis 1730.95 1798.33

# 3) render the 25-layer window along +n with vc_render_tifxyz (or villa infer.py --direction reverse), REVERSE it,
#    run the model, then score its map against the transferred label:
# python -m ink9face score --map my_map.tif --label out/${SEG}_w0_ink.tif --mask out/${SEG}_w0_mask.tif

# 4) optional two-face check when you have run the model on both orders:
# python -m ink9face face --mesh ink9face_cache/$SEG/coarse --map-native p_native.tif --map-reversed p_reversed.tif \
#        --mask out/${SEG}_w0_mask.tif
