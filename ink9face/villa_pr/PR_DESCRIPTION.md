# ink-detection: `--direction auto` / `--reverse_layers auto` for 9 µm inference

**Draft. Not opened.** Patch: `0001-ink-inference-direction-auto.patch`, against `ScrollPrize/villa` main at
`47a9c37` (apply from the repo root with `git apply villa_pr/0001-ink-inference-direction-auto.patch`).

## What it changes

`ink-detection/koine_machines/inference/infer.py` takes the layer order from `--direction forward|reverse|both`. The
default is `forward`, the order of the surface-volume zarr. For a `vc_render_tifxyz` render that order puts layer
index increasing along the VC3D normal +n (n = dP/dcol × dP/drow). No rule is applied, so a newly traced segment is
read in its native order unless someone passes `reverse`. The inference container works the same way: it reverses
only when `FORCE_REVERSE=true`.

The patch adds:

* `--direction auto`. It resolves once per segment, in both folder mode and single-zarr mode:
  1. If the segment's `metadata.json` / `meta.json` sets `reverse_layers`, that value wins. This file is in the
     segment directory, or next to the `.zarr`. If the value disagrees with the rule, a warning is logged.
  2. Otherwise the order is `reverse`, the anti-normal rule.

  The resolved order is logged. In folder mode it also appears in the output name (`..._reverse_<date>.tif`).
* `--reverse_layers {auto,true,false}` (also spelled `--reverse-layers`). It is shorthand for
  `--direction auto|reverse|forward`. Combining it with a different explicit `--direction` is an error.
* `tests/test_direction_auto.py`: 10 tests, CPU only, no data. They cover the default, metadata override (bool or
  string), URL inputs, the existing directions and the CLI shorthand.

The default stays `forward`, so existing jobs behave exactly as before. Whether `auto` should become the default is
for the maintainers to decide, and the evidence below is meant to inform that decision.

## Why: measured on public data

We ran public 9 µm checkpoints on both layer orders of 1 cm windows of organizer tifxyz meshes, on 5 public
scrolls: PHerc0500P2, PHerc0814, PHerc0343P, PHerc0009B and PHerc0139. Each model was scored only on segments it was
not trained on, using its public training lists. The reference labels are the organizers' 2.2-2.4 µm ink-detection
outputs, transferred onto the 8.64 / 9.362 µm mesh by an exact, geometry-only rule (no model sees them). Full
method, data list and scripts: the ink9face package (`eval/reproduce.py` rebuilds the table from the public bucket).

**How often `reverse` (anti-normal) reads the inked face:**
* 41/43 windows against clean-model truth;
* 97/99 windows against the TRACE oracle (all 5 scrolls);
* `forward` (the current default): 2/43.

**Held-out pixel AUC, forward → reverse:**

| model | windows / meshes | AUC forward → reverse | Δ AUC [95 % mesh-bootstrap CI] |
|---|---|---|---|
| ink_9um (seeds 42+43) | 43 / 36 | 0.515 → 0.619 | +0.099 [+0.069, +0.129] |
| KLAVIS ink9um-dense | 43 / 36 | 0.517 → 0.674 | +0.151 [+0.122, +0.181] |
| v8-in | 34 / 27 | 0.490 → 0.715 | +0.221 [+0.189, +0.253] |
| Reader v3 (PHerc0814) | 6 / 3 | 0.546 → 0.766 | +0.220 [+0.156, +0.258] |
| Reader v2 (PHerc0500P2) | 10 / 10 | 0.509 → 0.769 | +0.260 [+0.189, +0.324] |
| Reader v3 (PHerc0500P2) | 10 / 10 | 0.516 → 0.759 | +0.243 [+0.169, +0.308] |
| Nieuwlaar dense_native (PHerc0500P2) | 10 / 10 | 0.495 → 0.699 | +0.204 [+0.137, +0.267] |
| d9v4C (1 clean window) | 1 / 1 | 0.535 → 0.900 | +0.365 |

The rule is a convention of the organizers' meshes and of the models trained on them, not a law of physics. A mesh
from another tracer, or one whose grid is transposed or mirrored, flips +n. Per-segment metadata therefore keeps
precedence, and `--direction both` remains available for checking a new tracer.

## Test plan

* `pytest ink-detection/koine_machines/inference/tests/test_direction_auto.py ink-detection/koine_machines/inference/tests/test_infer.py`
  passes on CPU (11 passed). Run against main `47a9c37` with the patch applied.
* Not yet done, and wanted from a maintainer: one GPU run of `infer.py --folder <segments> --direction auto` on a
  segment whose metadata sets `reverse_layers`, and one on a segment without it, checking the log line and the
  output name.
* Possible follow-up, not in this patch: make the container honour `REVERSE_LAYERS=auto` (`entrypoint.py`, next
  to `FORCE_REVERSE`).
