"""WP9: whole-scroll flatten (anchors, coarse + tiles) and chart-space render / warp / cut."""
import numpy as np
import pytest

from unroller.flatten import flatten
from unroller.phantom import SpiralScroll
from unroller.surface import Surface
from unroller.volume import ArrayVolume
from unroller.ws.wflatten import (VIRTUAL, FusedGrid, coarse_stride, flatten_coarse, flatten_tiled, upsample,
                                  window_procrustes)
from unroller.ws.wrender import check_handedness, cut_wraps, direct_uv_render, ncc, render_chart, warp_to_uv


@pytest.fixture(scope="module")
def spiral():
    P = SpiralScroll()
    S = P.surface(step=1.0, distort=0.0)
    return P, S, flatten(S)


def test_defaults_are_the_plain_flatten(spiral):
    _, S, G = spiral
    U = flatten(S, init_uv=None, anchor_mask=None, anchor_uv=None, anchor_weight=1.0, vertex_weight=None)
    assert np.array_equal(U, G, equal_nan=True)


def test_anchored_tile_reproduces_global(spiral):
    _, S, G = spiral
    r0, r1, c0, c1, m = 20, 68, 200, 600, 16
    sub = Surface(S.points[r0 - m:r1 + m, c0 - m:c1 + m], S.scale)
    Gw = G[r0 - m:r1 + m, c0 - m:c1 + m]
    U = flatten(sub, init_uv=Gw, anchor_mask=sub.valid, anchor_uv=Gw, anchor_weight=1.0)
    assert np.nanmax(np.linalg.norm(U - Gw, axis=-1)[m:-m, m:-m]) <= 0.05


def test_mirrored_anchors_are_respected(spiral):
    # anchors given in the mirrored frame: the solution follows them (no reflection fight, no distortion)
    _, S, G = spiral
    sub = Surface(S.points[10:60, 100:400], S.scale)
    Gw = G[10:60, 100:400] * np.array([1.0, -1.0])
    U = flatten(sub, init_uv=Gw, anchor_mask=sub.valid, anchor_uv=Gw)
    assert np.nanmax(np.linalg.norm(U - Gw, axis=-1)) <= 0.05


def test_coarse_stride_and_upsample_cover_the_edges():
    assert coarse_stride((88, 1479), (11, 185)) == 8
    c = np.stack(np.meshgrid(np.arange(185) * 8.0, np.arange(11) * 8.0), -1)
    up = upsample(c, (88, 1479), 8)
    rr, cc = np.mgrid[0:88, 0:1479]
    assert np.allclose(up, np.stack([cc, rr], -1))  # linear map: exact, including the last rows / columns


def test_tiled_matches_global_with_small_seams(spiral):
    P, S, G = spiral
    fg = FusedGrid(S.points, np.zeros(S.shape, np.float32), chart={"step": 1.0})
    cu, ci = flatten_coarse(fg, step=8)
    UV, ti = flatten_tiled(fg, cu, k=ci["stride"], tile=128, margin=16)
    assert ti["n_tiles"] > 1 and ti["flips"] == 0 and ti["all_converged"]
    assert ti["seam_p95"] <= 0.5
    w = window_procrustes(UV, G, S.valid, 32, 1.0)
    assert np.percentile(w[:, 0], 95) <= 0.3
    wt = window_procrustes(UV, P.unrolled_coords(S.points), S.valid, 32, 1.0)
    assert np.percentile(wt[:, 0], 95) <= 0.3


def test_chart_render_warp_and_mirror_control(tmp_path):
    P = SpiralScroll()
    V = ArrayVolume(P.volume())
    S = P.surface(step=2.0, distort=0.0)
    fg = FusedGrid(S.points, np.zeros(S.shape, np.float32), chart={"step": 2.0})
    uv = flatten(fg.to_surface())
    r = render_chart(fg, V, str(tmp_path), n_layers=21, scale=2.0)
    assert r["shape"][1] > 2000
    w, info = warp_to_uv(str(tmp_path / "mean.npy"), fg, uv, scale=1.0, chart_scale=2.0)
    dm, dx = direct_uv_render(fg, V, uv, origin=info["origin"], shape=tuple(info["shape"]))
    assert ncc(w, dm)[0] >= 0.98
    assert abs(ncc(w, dm[:, ::-1])[0]) <= 0.10  # mirrored-ink control
    assert check_handedness(fg, outward_xyz=P.center) > 0.99


def test_cut_wraps_and_fused_grid_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(20, 90, 3)).astype(np.float32)
    H = np.broadcast_to((np.arange(90) // 30 + (np.arange(90) % 30) / 30.0)[None], (20, 90)).astype(np.float32)
    status = np.zeros((20, 90), np.uint8)
    status[5:8, 40:50] = VIRTUAL
    fg = FusedGrid(xyz, H, status=status, chart={"step": 8.0})
    fg.save(tmp_path / "fg.npz")
    fg2 = FusedGrid.load(tmp_path / "fg.npz")
    assert np.array_equal(fg2.xyz, fg.xyz) and np.array_equal(fg2.status, fg.status) and fg2.step == 8.0
    uv = np.stack(np.meshgrid(np.arange(90) * 8.0, np.arange(20) * 8.0), -1)
    cuts = cut_wraps(fg, uv, str(tmp_path / "wraps"))
    assert sorted(cuts) == [0, 1, 2]
    for w, c in cuts.items():
        assert c["cols"] == [30 * w, 30 * w + 30]
        assert np.array_equal(c["surface"].points, xyz[:, 30 * w:30 * w + 30])
    assert np.isnan(fg.to_surface().points[6, 45, 0]) and not np.isnan(fg.to_surface(exclude=None).points[6, 45, 0])
