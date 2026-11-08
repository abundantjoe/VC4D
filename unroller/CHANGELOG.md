# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [0.2.0] - unreleased (first public release)

The first public version. Earlier versions were internal and are not published.

### Added

- `unroller trace`: grows a tifxyz sheet from one seed point on a remote or local OME-Zarr. It uses a mean-shift
  snap along the normals and a plane-fit front, with wrap-jump safeguards (default, "C2"). `trace.LEGACY` restores
  the previous tracer.
- `unroller flatten`: grid-initialised multiresolution ARAP (`--init lscm` is also available), `--downsample k`
  for large meshes, and a distortion report (singular values, flips, area ratio) written next to the output.
- `unroller render`: renders N layers along the normals (the `vc_render_tifxyz` offset convention) to `NN.tif` and
  `layers.zarr`, with numba-compiled trilinear kernels. The output is bit-identical between the cold remote,
  disk-cache, RAM-cache and in-memory paths.
- `unroller.volume`: OME-Zarr over HTTPS with parallel chunk fetches, a RAM LRU and an optional disk cache.
- `unroller obj2tifxyz`, `info`, `refine`, `regularize`, `phantom`, `validate`, `bench`, `crackle`, `budget`.
- `unroller.ws.solve`: an exact integer L1 winding-number assignment by max-flow, with an optimality certificate.
  It uses PyMaxflow when it is installed (`pip install .[bk]`) and scipy otherwise.
- Benchmarks on public Scroll 1 data (`benchmarks/`), with raw JSON results and the script that makes the README
  figures (`benchmarks/scripts/make_figs.py`, output in `docs/figs/`).
- `scripts/quickstart_scroll1.sh`: the README quick start as one script.
- A GitHub Actions workflow that runs `unroller validate` and the test suite on Python 3.10 and 3.12.
- `CITATION.cff`, this changelog and the MIT licence.

### Fixed

- Remote stores (OME-Zarr and TIFF stacks) now retry a transfer cut off mid-body (`ChunkedEncodingError`),
  connection resets, read timeouts and 429/5xx answers with exponential backoff. Before, one dropped connection
  lost a whole `--volume-16bit` render. `UNROLLER_HTTP_RETRIES` sets the number of retries (default 5).
  `tests/test_tiffstack.py::test_http_stack_retries_dropped_transfers` serves a TIFF stack through a local server
  that cuts off or resets every third response and checks the data are still exact.

### Documented

- The ink end-use check with the 2023 Grand Prize model (`benchmarks/ink_gp/`): 16-bit renders match the official
  layers, uint8-zarr renders lose about 0.05 AUC.

### Removed (compared with the internal versions)

- The retired physics-cue modules (`physink`, `texbank`) and the `physink` CLI command.
- Internal bookkeeping (provenance ledger, preregistration files, fleet and sync scripts).

### Known limitations

- No head-to-head comparison with the C++ VC3D tools (`vc_render_tifxyz`, `vc_flatten`,
  `vc_grow_seg_from_seed`) yet. See "What is not claimed yet" in the README.
