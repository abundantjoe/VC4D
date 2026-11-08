# ink9face: read 8-9 µm ink on the right face, with exact labels

Small, model-agnostic tools for 8-9 µm ink detection on Vesuvius Challenge `tifxyz` meshes:

1. **Face / layer order.** Ink models with a 3D stem are not symmetric in depth. Feed them the 25-layer stack in the
   wrong order and they read the other face of the sheet, which puts AUC near chance. Our measurements on organizer
   meshes of **5 public scrolls** show one label-free rule that holds: **reverse the stack relative to the VC3D
   normal** (`reverse_layers=True`). It picks the inked face on **41/43** windows against clean-model truth and
   **97/99** against the TRACE oracle. Across models, the AUC gain over the renderer default is **+0.10 to +0.22**
   (ink_9um, KLAVIS, v8-in, Reader v3), and **+0.20 to +0.26** for the newest public checkpoints (Reader v2,
   Reader v3, dense_native, on 10 clean PHerc0500P2 meshes). The headline table can be rebuilt from the public bucket
   (`eval/reproduce.py`).
2. **Exact geometry-only label transfer.** Organizer ink labels at 2.2-2.4 µm are mapped onto the same segment's
   8.64 / 9.362 µm mesh with a closed-form grid rule, checked against the published inter-volume affine. No image
   registration is used and no model sees the labels, so the transfer is not circular. **106/106** windows pass with a
   3D residual p95 of at most 0.47 voxels.
3. **Depth-convention check.** A label-free CT-profile offset between two mesh conventions, plus a measured depth
   sweep that shows what a wrong offset costs.
4. **Metrics.** Pixel AUC, edge-tolerant AUC, a 37 µm block AUC that stays comparable across resolutions, and a
   letter-scale correlation with a shift null. Bootstrap CIs resample meshes, not windows.

Licence: MIT (code). The data used in the evaluation is Vesuvius Challenge open data
(`s3://vesuvius-challenge-open-data`). Labels derived from it fall under the open-data terms (non-commercial,
attribution) and are not redistributed here. Use `ink9face labels` to regenerate them.

## The problem

At 8-9 µm the ink signal is a few voxels thick, so it sits on one face of the papyrus sheet. A segment mesh has two
faces, and the order of the rendered layers decides which one a 3D-stem model reads. villa's ink inference
(`ink-detection/koine_machines/inference/infer.py`, main at `47a9c37`) takes that order from
`--direction forward|reverse|both`. The default is **`forward`**, the native order of the surface volume. The
inference container reverses only when `FORCE_REVERSE=true`. Neither applies a rule, so a new segment is read in its
native order unless someone already knows better. Our own earlier pipeline used a third rule, "read toward the
scroll axis", which has no meaning on a flat fragment. `villa_pr/` holds a draft patch that adds
`--direction auto` / `--reverse_layers auto` using the rule below.

## Method

* `+n` = normalize(dP/dcol × dP/drow), the VC3D convention (`Geometry.cpp grid_normal`). An independent
  implementation agrees with it (cos = 1.000 on a PHerc0500P2 mesh). `vc_render_tifxyz` writes slice *k* at offset
  (k − (n−1)/2)·step along +n, so its **native** stack has the layer index increasing along +n.
* **anti_normal rule:** feed `native[::-1]`. The rule is fixed and has no parameter. It is a measured convention of the
  organizer meshes and models, not a physical law (see Limitations).
* Fallback for meshes whose normal sign you do not trust: run the model on both orders and compare the letter-scale
  band energy of logit(p) (`face.choose_band`). Sum it over a whole scroll (`face.scroll_vote`) before trusting it.
* **Label transfer:** for coarse grid point g9, label grid point g2 + 0.5 = (H2/H9, W2/W9)·(g9 + 0.5), using the exact
  per-segment grid ratios. Label pixel = round(20·g2). Validation: |T·P9(g9) − P2(g2)| at the nodes, where T is the
  published `transform.json` or, if none is published, a trimmed LSQ affine. A window passes if the median is below 1
  voxel and p95 is below 2 voxels.

## Usage

```bash
pip install -e .            # numpy, scipy, tifffile, requests (matplotlib only for figures)

# exact labels for a window of a public segment (downloads the label mesh, label tif, coarse mesh, transform)
ink9face labels --scroll PHerc0500P2 --segment 20250611132918-z_dbg_gen_00302_inp_hr \
                --crop 152 10 53 53 --out out/seg_w0
#  out/seg_w0_ink.tif  (uint8, 1 px = 1 voxel of the 9.362 um scan), out/seg_w0_mask.tif, out/seg_w0.json (gate, T, residuals)
# same from local files:
ink9face labels --coarse-mesh MESH9.tifxyz --label-mesh MESH2.tifxyz --label-tif label.tif \
                --transform transform.json --label-um 2.215um --crop 152 10 53 53 --out out/seg_w0

# layer order for a mesh (optionally compared with the toward-axis rule, or with a two-face model check)
ink9face face --mesh MESH9.tifxyz --crop 152 10 53 53 --axis 1730.95 1798.33
ink9face face --mesh MESH9.tifxyz --map-native p_nat.tif --map-reversed p_rev.tif --mask out/seg_w0_mask.tif

# score any ink map against a transferred label
ink9face score --map p.tif --label out/seg_w0_ink.tif --mask out/seg_w0_mask.tif

# depth-convention offset between two sets of rendered stacks (.npz with st (L,H,W) and valid (H,W))
ink9face depth --ref benchmark_stacks/ --tgt my_stacks/
```

In Python: `from ink9face.face import orient; x = orient(native_stack)` returns the stack in model order.
`examples/run_example.sh` runs steps 1-2 end to end on one public PHerc0500P2 segment. The derived label window
it produces is not shipped here (open-data terms); run the script to regenerate it.

Formats: meshes are `tifxyz` directories (`x.tif y.tif z.tif`, invalid nodes −1). Labels and masks are uint8 TIFFs on
the mesh render grid (20 px per cell = 1 px per voxel at scale 0.05). Maps are any float [0,1] or uint8 image of the
same grid. The volumes that the transform links are the organizers' OME-Zarr scans.

## Evaluation (public data only)

### Reproduce from public data (no private files)

```bash
pip install -e ".[dev]"
# 1. labels: downloads label mesh, 2.2-2.4 um label tif, coarse mesh and transform per segment from
#    s3://vesuvius-challenge-open-data (anonymous HTTPS, ~3 GB in total, one segment on disk at a time),
#    transfers the label onto the 99 windows in eval/windows.json and gates each window in 3D
python eval/reproduce.py labels --cache ink9face_cache
# 2. table: scores the two-face model maps against those labels, excluding every model on the segments it was
#    trained on (eval/exclusions.json, from the public training lists), and checks the result
python eval/reproduce.py table --cache ink9face_cache --maps MAPS --out results/face_eval_public_repro.json \
       --check results/face_eval.json --tol 2e-3
```

`MAPS` holds one file per model and window, `<model>/<window>.npz`, with arrays `native` and `anti_normal` (layout
in `eval/face_table.py`). It comes from one of two places:
* the maps bundle (431 MB, 266 map pairs; available on request, or rebuild it with `eval/export_maps.py`);
* `eval/infer_two_faces.py`, which runs any public 9 µm checkpoint on both layer orders of a rendered window
  through villa's own model loader (`koine_machines`).

`eval/export_maps.py` documents how the bundle was cut from our cache. It renames and re-orients the arrays and
leaves the values unchanged.

**Reproduction check (2026-11-08, `results/reproduce_check.json`).**
* **Labels.** All 99 windows were re-downloaded and re-transferred, and all passed the gate. On the 88 windows whose
  evaluation labels were still on disk, ink and label support are identical pixel for pixel.
  * The mesh-validity mask differs on at most 0.06 % of the scored pixels. The evaluation used the renderer's
    validity mask; the public path uses bilinear validity of the coarse mesh.
* **Table.** Rebuilt from public labels plus the bundle on the 88 windows of 4 scrolls:
  * every count is identical (41/43, 32/43, 27/29, 14/14, …), and so is the truth on every window (88/88);
  * the largest per-window AUC / letter-r difference is 0.0007;
  * every summary number agrees within 1e-3, with one exception: a scroll band-vote log sum, 1.1986 vs 1.1975.
* **Gap.** The 11 PHerc0139 windows (TRACE oracle only) are not in the bundle, because their cached maps were
  deleted from our disk before the export. Their labels are reproduced; regenerating their maps needs
  `infer_two_faces.py`. Until then, `--check` against the full `face_eval.json` also lists the
  PHerc0139 entries and the TRACE `ALL` pools as differences; the 88-window comparison is in
  `results/reproduce_check.json`.

The other scripts (`eval/eval_labels.py`, `eval/eval_ct_cue.py`, `eval/make_figs.py`) still read our cache tree
(`--scratch`). The CT-cue study needs the rendered CT stacks, which are not redistributed.

**Data.** Organizer meshes and organizer 2.2-2.4 µm ink-detection outputs for PHerc0500P2 (29 windows / 29 meshes),
PHerc0814 (14 / 7; segment 46527 excluded because every model saw it), PHerc0343P (15 / 8; 8.64 µm), PHerc0009B
(30 / 15; 8.64 µm) and PHerc0139 (11 / 7). A window is 54×54 mesh cells = 1061×1061 px ≈ 1 cm². None of these scrolls is
First-Letters-eligible. **Note:** the reference "labels" are the organizers' **high-resolution ink-detection
predictions** (for example `...-new_canon_autoresearch_recipe-tile256-stride128.tif`), thresholded at 127. They are
not hand annotations.

**Models** (public checkpoints): the organizers' ink_9um (mean of seeds 42 and 43), KLAVIS, v8-in, Reader v3 and TRACE
(sections 1-2). Section 2b adds Reader v2, Reader v3 on PHerc0500P2, d9v4C and dense_native. Every model is excluded on segments it was trained on (`eval/exclusions.json`, built from the public
training lists and model cards; sources are listed in the file). TRACE was trained
on everything, so it serves only as a face oracle where no clean model has both faces (0343P, 0009B, 0139).

### 1. Face rule accuracy (fig2)

| truth | anti_normal | native (+n) | toward axis | two-face band (KLAVIS) |
|---|---|---|---|---|
| clean models, PHerc0500P2 | **27/29** | 2/29 | 6/29 | 21/29 |
| clean models, PHerc0814 | **14/14** | 0/14 | 14/14 | 11/14 |
| TRACE oracle, all 5 scrolls | **97/99** | 2/99 | 62/99 | – |

The four exceptions are near-ties or single windows. On the clean side: 2 PHerc0500P2 windows with clean AUC gap
−0.047 and −0.060. Against TRACE: 1 window of 0343P and 1 of 0009B. As a whole-scroll vote, the two-face band check
agrees with anti_normal on 0500P2 and 0814 for KLAVIS and ink9avg. For TRACE it agrees on 4/5 scrolls (0814 is a tie).

### 2. What the face is worth (fig1; per-mesh paired differences, 10,000-draw mesh bootstrap, one-sided sign test)

| model (clean windows) | windows / meshes | AUC native → anti_normal | Δ AUC [95 % CI] | meshes better | letter r native → anti | AUC toward-axis → anti: Δ |
|---|---|---|---|---|---|---|
| ink_9um (42+43) | 43 / 36 | 0.515 → 0.619 | **+0.099** [+0.069, +0.129] | 32/36 | 0.005 → 0.071 | +0.059 [+0.032, +0.089] |
| KLAVIS | 43 / 36 | 0.517 → 0.674 | **+0.151** [+0.122, +0.181] | 34/36 | 0.006 → 0.111 | +0.096 [+0.063, +0.132] |
| v8-in | 34 / 27 | 0.490 → 0.715 | **+0.221** [+0.189, +0.253] | 27/27 | 0.001 → 0.112 | +0.158 [+0.111, +0.205] |
| Reader v3 (0814 only) | 6 / 3 | 0.546 → 0.766 | +0.220 [+0.156, +0.258] | 3/3 | −0.011 → 0.178 | 0 |
| TRACE (in-sample) | 99 / 66 | 0.544 → 0.738 | +0.198 [+0.172, +0.222] | 64/66 | 0.022 → 0.175 | +0.094 [+0.064, +0.125] |

The 37 µm block AUC gives the same differences to ±0.001 (`results/face_eval.json`). anti_normal reaches the
per-window oracle to within 0.005 AUC for every model.

Against the **toward-axis** rule the gain is limited to PHerc0500P2 (a fragment) and the 8.64 µm scans. On PHerc0814 and
PHerc0139 the two rules coincide, so the gain there is 0.

### 2b. Newest public checkpoints (PHerc0500P2 clean windows, `results/face_eval_newck.json`)

We ran Reader v2, Reader v3, d9v4C and Nieuwlaar dense_native on both layer orders of 10 PHerc0500P2 windows.
* **Run settings:** `eval/infer_two_faces.py`, villa loader, input slab centred at shift 0, CPU, bf16.
* **Clean windows:** none of the 10 is in the training lists of Reader v2/v3 or dense_native
  (`eval/exclusions.json`). d9v4C was fine-tuned on every other PHerc0500P2 segment of the evaluation set, so only
  1 window is clean for it.
* **Labels:** the public-path labels (`reproduce.py labels`).
* **Why fresh runs:** the cached maps in our earlier sweep held only one face at shift +2, so they cannot give a
  paired effect. On the 10 windows that sweep covered for dense_native, its mean AUC was 0.718, against 0.699 for
  our anti_normal map at shift 0 (per-window r = 0.96).
* **Numerics check:** our native Reader v3 map on window `20250825181859--1_w0` matches the earlier
  public-inference map at r = 0.9998.

| model (clean windows) | windows / meshes | AUC native → anti_normal | Δ AUC [95 % CI] | meshes better | letter r native → anti |
|---|---|---|---|---|---|
| Reader v2 | 10 / 10 | 0.509 → 0.769 | **+0.260** [+0.189, +0.324] | 10/10 | 0.023 → 0.239 |
| Reader v3 | 10 / 10 | 0.516 → 0.759 | **+0.243** [+0.169, +0.308] | 9/10 | 0.014 → 0.239 |
| dense_native | 10 / 10 | 0.495 → 0.699 | **+0.204** [+0.137, +0.267] | 10/10 | 0.025 → 0.142 |
| d9v4C (1 clean window) | 1 / 1 | 0.535 → 0.900 | +0.365 (n = 1) | 1/1 | 0.034 → 0.386 |

* **Face picked:** with these models as truth, anti_normal picks the better face on 10/10 windows. The toward-axis
  rule picks it on 2/10.
* **Gain over toward-axis:** anti_normal minus toward-axis is +0.19 to +0.23 AUC.
* **Oracle:** anti_normal equals the per-window oracle for every model except Reader v3, which is within 0.001 of
  it on average.
* **Block AUC:** the 37 µm block AUC gives the same Δ to ±0.002.
* **Effect size:** all 4 newest checkpoints show the effect, and it is larger than for the older models
  (+0.20 to +0.26 vs +0.10 to +0.22).
* **Run cost:** 62 CPU inferences of about 160 s each.

### 3. Exact labels vs the previous image-fit labels (PHerc0500P2, 29 windows, correct face)

* Registration (fig5): all 106 windows of the 5 scrolls pass. Node residual p95 is at most 0.23 voxels (0500P2),
  0.37 (0814, LSQ affine), 0.46 (0343P), 0.47 (0009B) and 0.28 (0139, LSQ). Where a transform is published, the
  independent LSQ estimate agrees within 0.06 voxels (p95).
* Effect: 1.5 % of pixels change label (median; max 7 %). KLAVIS AUC 0.659 → 0.663, Δ +0.005 [−0.001, +0.012].
  Letter r +0.007 [+0.000, +0.015]. ink_9um Δ AUC −0.001 [−0.005, +0.003]. **For evaluation this is a null-to-tiny
  gain:** a good IoU image fit was already nearly right. The value of the exact transfer is that it is checkable (a
  3D gate per window), needs no ds8 image or search, and can be used for training labels without circularity.

### 4. Depth (fig4, PHerc0500P2, correct face)

AUC and letter r are flat for input-window shifts of 0..+3 voxels and fall for negative shifts (KLAVIS AUC 0.663 at 0,
0.615 at −3; Reader letter r 0.210 → 0.137). A convention that sits 3 voxels too shallow costs about 0.03-0.05 AUC. This
curve comes from an earlier sweep (`eval_shift.json`, computed by the same metrics); this package does not re-run it.

### 5. Negative results, reported as such

* **CT-only face cues fail** (`results/ct_cue.json`): density asymmetry, density centroid and texture asymmetry along
  the normal, on 40 public windows. None separates the faces (positive-sign share 0.38-0.58). The two native-right
  windows are the ones with the sheet strongly on the −n side, but 2 cases cannot support a rule.
* **Per-window model statistics are weak:** the two-face band rule gets 32/43 windows. An earlier search found the best
  label-free statistic at 20/29 on 0500P2. Curvature ("read the concave side") gets 6/15.
* **Exact labels do not raise AUC measurably** (section 3).

## Limitations

* anti_normal is an **empirical convention** of the organizers' meshes plus the public 9 µm models trained on them, on
  5 scrolls and 5 model families. A mesh made by another tracer, or with its grid transposed or mirrored, flips +n.
  Check those with the two-face band vote over many windows, or with a few labelled windows, before trusting the rule.
* The truth is model-based (label-scored AUC of clean models, or TRACE). The labels are organizer model outputs at
  2.2 µm, not human annotation.
* Small clusters: in the headline table Reader v3 is clean on only 3 PHerc0814 meshes. The newest checkpoints are
  measured on 10 PHerc0500P2 meshes only, and d9v4C on a single window. 0343P, 0009B and 0139 have only the TRACE
  oracle.
* The public reproduction covers 88 of the 99 windows. The PHerc0139 TRACE maps were lost before the bundle was cut.
* No model weights ship here. `eval/infer_two_faces.py` runs public checkpoints through villa's own loader
  (`koine_machines`), so it needs a villa clone. Related, not included: a leave-one-scroll-out fine-tune of KLAVIS on
  these labels transferred on 2 of 4 held-out scrolls (PHerc0009B +0.028 AUC [+0.019, +0.036], 15/15 meshes;
  PHerc0500P2 +0.014; PHerc0343P and PHerc0814 null).

## What would make it stronger

* Getting `villa_pr/` (`--direction auto` / `--reverse_layers auto`) reviewed and merged. The patch, tests and
  description are ready; it has not been opened.
* Hand labels on a few windows per scroll, to replace model-based truth.
* Meshes from a second tracer (for example the ScrollFiesta/spiral output), to test the convention beyond organizer
  meshes.
* The newest checkpoints on more scrolls. On PHerc0814, 0343P and 0009B their training lists leave few clean meshes.

## Tests and CI

`python -m pytest` runs 16 tests on synthetic data and the shipped lists. No downloads are needed. The tests cover:
* VC normals, orient, the label rule, affine fit and metrics;
* a synthetic two-grid label transfer;
* face-table scoring, contamination handling and the `reproduce.py table` round trip;
* window and exclusion list integrity, plus a check that no First-Letters-eligible or exam scroll id appears.

The VC4D workflow (`../.github/workflows/ci.yml`) runs them on Python 3.10 and 3.12, plus a CLI smoke test.

## Layout

```
ink9face/   tifxyz.py (load, VC normals, bilinear, projection)  labels.py  face.py  depth.py  metrics.py  cli.py
eval/       reproduce.py (public end-to-end)  face_table.py  windows.json  exclusions.json
            infer_two_faces.py (villa loader, both orders)  export_maps.py (cache -> maps bundle)
            eval_labels.py  eval_ct_cue.py  make_figs.py
results/    face_eval.json (+ _rows, .log)  face_eval_public_repro.json  reproduce_check.json  face_eval_newck.json
            labels_eval.json  ct_cue.json
figs/       fig1..fig5 (public scrolls only)
villa_pr/   0001-ink-inference-direction-auto.patch  PR_DESCRIPTION.md  (draft, not opened)
examples/   run_example.sh (regenerates one transferred public label window)
tests/      test_core.py  test_eval.py
```

## Citations and data

Data citations (Vesuvius Challenge CT scans), model credits and licences: [../CITATIONS.md](../CITATIONS.md). The
scroll data and the figures derived from it are CC BY-NC 4.0; the code is MIT.
