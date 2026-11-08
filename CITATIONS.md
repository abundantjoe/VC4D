# Data, models and code used in VC4D: citations and credits

The rules this follows are at <https://scrollprize.org/data> ("How to Cite", "Licenses") and
<https://scrollprize.org/prizes>.

## Scroll data

All scroll data is Vesuvius Challenge open data, licensed **CC BY-NC 4.0**
(<https://creativecommons.org/licenses/by-nc/4.0/>). The figures, the small test fixture and the benchmark
results in this repository are derived from that data, so they stay under CC BY-NC 4.0. The code is MIT.

### Scroll 1 (PHerc. Paris 4): used by `unroller`

Scroll 1 comes from the **EduceLab-Scrolls** dataset, copyrighted by EduceLab / The University of Kentucky, and is used
under the permission granted to Vesuvius Challenge.

> Data used in the preparation of this work were obtained from the EduceLab-Scrolls dataset [1].

[1] Parsons, S., Parker, C. S., Chapman, C., Hayashida, M., & Seales, W. B. (2023). EduceLab-Scrolls: Verifiable
Recovery of Text from Herculaneum Papyri using X-ray CT. *ArXiv [Cs.CV]*. https://doi.org/10.48550/arXiv.2304.02084

```bibtex
@misc{parsons2023educelab,
  author = {Parsons, Stephen and Parker, C. Seth and Chapman, Christy and Hayashida, Mami and Seales, W. Brent},
  title  = {{EduceLab-Scrolls}: Verifiable Recovery of Text from {Herculaneum} Papyri using {X}-ray {CT}},
  year   = {2023},
  eprint = {2304.02084},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi    = {10.48550/arXiv.2304.02084}
}
```

### PHerc0500P2, PHerc0814, PHerc0343P, PHerc0009B, PHerc0139: used by `ink9face`

These are newer scans released by Vesuvius Challenge. Cite:

[2] Angelotti, G., Parsons, S., Johnson, S., Dal Prà, E. R., Rudolph, J., Tafforeau, P., Mirone, A., Henderson, P.,
Schilling, H., McDonald, F., Josey, D., Nader, Y., Parker, C. S., & Seales, W. B. *Vesuvius Challenge - CT Scans of
Herculaneum Papyri*. Vesuvius Challenge. https://scrollprize.org/data

The ink labels used for evaluation are the organizers' public high-resolution (2.2-2.4 um) ink outputs on the same
scrolls, transferred by geometry onto the 9 um meshes. They are used as labels only, never as model input.

## Models (credited; no weights are redistributed here)

| model | used for | source | licence |
|---|---|---|---|
| 2023 Grand Prize TimeSformer | unroller ink end-use check | Youssef Nader, Luke Farritor, Julian Schilliger: <https://huggingface.co/scrollprize/timesformer_GP_scroll1>, <https://github.com/younader/Vesuvius-Grandprize-Winner> (also the GP hand labels of segment 20230531121653) | MIT |
| ink_9um (seeds 42, 43) | ink9face evaluation | Vesuvius Challenge: <https://huggingface.co/scrollprize/ink_9um> | MIT |
| KLAVIS dense9um_all7 | ink9face evaluation | Domenico Russo: <https://huggingface.co/domenicor046/ink9um-dense> | see model card |
| Reader v2, Reader v3 | ink9face evaluation | Domenico Russo: <https://huggingface.co/domenicor046/reader-v2>, <https://huggingface.co/domenicor046/reader-v3> | MIT |
| dense_native_016000 | ink9face evaluation | Erwin Nieuwlaar: <https://huggingface.co/Nieuwlaar/ink9um-dense-native> | MIT |
| d9v4C | ink9face evaluation (1 clean window) | TAUIL-Abd-Elilah: <https://github.com/TAUIL-Abd-Elilah/9um-reader-letter-shape> | code MIT, data CC BY-NC |
| v8-in | ink9face evaluation | Youssef Nader: <https://huggingface.co/YoussefMoNader/ink-8um-v8in> | MIT |
| TRACE v3B | ink9face face oracle only | <https://github.com/Jashann/vesuvius-scrolling> | see repository |

Each model is scored only on windows from segments it was not trained on (`ink9face/eval/exclusions.json`).

## Software

- **villa / VC3D** (ScrollPrize): <https://github.com/ScrollPrize/villa>. `koine_machines` loads the 9 um ink models in
  `ink9face/eval/infer_two_faces.py`. `vc_render_tifxyz` (volume-cartographer @ 2144fdd) is the baseline in the
  unroller head-to-head. `ink9face/villa_pr/` is a draft patch against villa.
- **libigl** (ARAP baseline), **PyTorch**, **SciPy**, **numba**, **numpy**, **tifffile**, **zarr**: the
  baselines and libraries named in the unroller benchmarks.

## Disclosure

The code and write-ups were developed with AI coding assistance (Claude).
