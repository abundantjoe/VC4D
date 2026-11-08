#!/bin/bash
# End-to-end profile of the CLI pipeline on PHerc. Paris 4 segment 20230504093154 (wall, CPU, peak RSS,
# bytes fetched, cProfile top-10 per stage, each stage in a fresh process).
# usage: scripts/e2e_scroll1.sh WORKDIR RESULTS.json      (SKIP_FULL=1 skips the full-grid flatten)
set -e
W=$1; OUT=$2; HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg
U=$ROOT/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr
mkdir -p $W; [ -f $W/seg.obj ] || curl -sS -o $W/seg.obj $ROOT/paths/20230504093154/20230504093154.obj
P="python $HERE/e2e_profile.py $OUT"
$P obj2tifxyz -- obj2tifxyz $W/seg.obj $W/seg
$P info -- info $W/seg > /dev/null
$P refine -- refine $W/seg $U $W/seg_ref --disk-cache $W/cacheA
if [ -z "$SKIP_FULL" ]; then $P flatten_full -- flatten $W/seg_ref $W/flat; fi
$P flatten_ds4 -- flatten $W/seg_ref $W/flat_ds4 --downsample 4
$P render_cold -- render $W/flat_ds4 $U $W/r_cold -n 65 --disk-cache $W/cacheB
$P render_warm -- render $W/flat_ds4 $U $W/r_warm -n 65 --disk-cache $W/cacheB
$P fibers $W/r_cold/layers.zarr
