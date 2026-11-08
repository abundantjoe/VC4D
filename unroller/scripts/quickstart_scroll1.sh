#!/bin/bash
# Quick start on public data: Scroll 1 (PHerc. Paris 4), segment 20230504093154, standardized 7.91 um OME-Zarr.
# Converts the official mesh, checks it, flattens it, renders a 512 x 512 x 65 layer crop along the normals and
# traces a new patch from a seed on the sheet.  About 3.6 MB of mesh plus about 35 MB of volume chunks are
# downloaded and about 250 MB is written.
# usage: bash scripts/quickstart_scroll1.sh [WORKDIR]      (default: ./quickstart)
set -euo pipefail
W=${1:-quickstart}; U=${UNROLLER:-unroller}
ROOT=https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg
VOL=$ROOT/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr
mkdir -p "$W"; cd "$W"
t0=$SECONDS; step() { echo "[$((SECONDS - t0)) s] $*"; }
step "download mesh";        [ -f seg.obj ] || curl -sSfo seg.obj $ROOT/paths/20230504093154/20230504093154.obj
step "obj2tifxyz";           $U obj2tifxyz seg.obj seg
step "info";                 $U info seg
step "flatten (every 4th vertex, UVs upsampled)"
                             $U flatten seg flat --downsample 4
step "render 65 layers, 512 x 512 crop"
                             $U render seg "$VOL" layers -n 65 --crop 384 2024 512 512 --workers 2 --disk-cache cache
step "trace 40 generations from a seed on the sheet"
                             $U trace "$VOL" traced --seed 3415.6 3753.4 10108.3 --generations 40 --disk-cache cache
step "done: seg/ flat/ layers/00.tif..64.tif traced/"
