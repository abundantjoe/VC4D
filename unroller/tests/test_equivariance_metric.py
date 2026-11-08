import numpy as np

from unroller import Surface
from unroller.metrics import (equivariance_switch, grid_rings, label_agreement, offsheet_vs_labels, sheet_crossing,
                              vertex_labels, wilson_ci)
from unroller.volume import ArrayVolume


def _plane(rows=30, cols=40, z=30.0, step=3.0, tilt=0.05, seed_rc=None):
    r, c = np.mgrid[0:rows, 0:cols]
    P = np.stack([c * step + 1.0, r * step + 1.0, z + tilt * c * step], -1)
    rc = seed_rc or (rows // 2, cols // 2)
    return Surface(P, (1 / step, 1 / step), {"seed": P[rc].tolist()})


def test_equivariance_switch_planted_patch_and_offset():
    ref = _plane()
    same = Surface(ref.points + [0.4, 0.7, 1.5], ref.scale)  # calibration offset, lateral resampling
    m = equivariance_switch(same, ref, 12.0)
    assert m["scored"] > 0.8 * 28 * 38 and m["switches"] == 0 and abs(m["offset_median_vx"] + 1.5) < 0.2
    assert not m["scored_map"][0].any() and not m["scored_map"][:, -1].any()  # outermost ring excluded
    Q = same.points.copy()
    Q[8:18, 10:22, 2] += 12.0  # wrap jump: patch on the neighbouring sheet one spacing away
    m = equivariance_switch(Surface(Q, ref.scale), ref, 12.0)
    assert m["switch_map"][9:17, 11:21].all()
    far = np.ones(Q.shape[:2], bool)
    far[6:20, 8:24] = False
    assert not (m["switch_map"] & far).any() and m["flagged"]
    hole = ref.points.copy()
    hole[5:25, 5:30] = np.nan  # no reference under the test: not scored, not a switch
    m = equivariance_switch(same, Surface(hole, ref.scale), 12.0)
    assert m["switches"] == 0 and not m["scored_map"][10:20, 10:25].any()


def test_equivariance_switch_test_outer_ring_excluded():
    big = _plane(36, 46, tilt=0.0)
    ref = Surface(big.points.copy(), big.scale)
    test = Surface(big.points[3:33, 3:43] + [0, 0, 1.5], big.scale)  # lies well inside the reference
    m = equivariance_switch(test, ref, 12.0)
    ring = np.zeros(test.shape, bool)
    ring[[0, -1]] = ring[:, [0, -1]] = True
    assert not m["scored_map"][ring].any() and m["scored_map"][~ring].all() and m["switches"] == 0


def test_equivariance_switch_reference_outer_ring_excluded():
    big = _plane(30, 40, tilt=0.0)
    ref = Surface(big.points[6:24, 6:34].copy(), big.scale)
    test = Surface(big.points + [0, 0, 1.5], big.scale)  # extends 6 cells beyond the reference on every side
    m = equivariance_switch(test, ref, 12.0)
    over = np.zeros(test.shape, bool)
    over[6:24, 6:34] = True
    rim = over.copy()
    rim[7:23, 7:33] = False  # cells over the reference's outermost ring: its newest, least relaxed vertices
    assert not m["scored_map"][rim].any() and m["scored_map"][7:23, 7:33].all() and not m["scored_map"][~over].any()


def test_equivariance_switch_cliff_and_spike_unscored():
    ref = _plane(30, 40, tilt=0.0)
    P = ref.points.copy()
    P[:, 20:, 2] += 12.0  # cliff: the right half of the reference is one sheet up
    P[10, 8, 2] += 30.0  # spike
    m = equivariance_switch(ref, Surface(P, ref.scale), 12.0)
    st = m["status_map"]
    # the nearest foot of a column 20-23 vertex is on the cliff wall (2.9 vx away at column 20), that of the
    # vertex under the spike on its flank 3 vx away: left unscored, not matched within the window
    assert (st[1:-1, 20:24] == 1).all() and st[10, 8] == 1 and (st[1:-1, 2:19] == 2).sum() == 28 * 17 - 1
    assert m["switch_map"][1:-1, 24:-1].all() and m["geom_unconfirmed"] == 0  # no evidence given: distance only


def _slabs(sheets, shape=(64, 64, 64)):
    """CT (200 on papyrus, 20 between) and labels; sheets = [(z0, z1, ct_on, label)] (z1 exclusive)."""
    ct, lab = np.full(shape, 20, np.uint8), np.zeros(shape, np.int32)
    for z0, z1, on, li in sheets:
        ct[z0:z1] = 200 if on else 20
        lab[z0:z1] = li
    return ArrayVolume(ct), lab


def test_neighbouring_sheet_test():
    ref = _plane(15, 18, z=27.0, tilt=0.0)
    up = lambda dz: Surface(ref.points + [0, 0, dz], ref.scale)  # noqa: E731
    # two wraps one spacing apart that share label ID 1 (an unlabelled, dark gap between them): a switch
    V, lab = _slabs([(24, 31, 1, 1), (37, 44, 1, 1)])
    m = equivariance_switch(up(13.0), ref, 12.0, volume=V, label_volume=lab)
    assert m["switch_geom"] == m["switches"] == m["scored"] > 100
    assert (m["cross_label_map"][m["scored_map"]] == 1).all() and (m["dip_map"][m["scored_map"]] < 0.2).all()
    # one thick sheet: the traces sit 8 vx (0.67 spacing) apart inside it -- a depth offset, not a switch
    V, lab = _slabs([(22, 40, 1, 1)])
    m = equivariance_switch(up(8.0), ref, 12.0, volume=V, label_volume=lab)
    assert m["switches"] == 0 and m["geom_unconfirmed"] == m["scored"] > 100 and (m["status_map"][m["scored_map"]] == 5).all()
    assert equivariance_switch(up(8.0), ref, 12.0)["switches"] == m["scored"]  # distance-only rule counts them
    assert equivariance_switch(up(8.0), ref, 12.0, volume=V)["switches"] == 0  # CT alone: no dip
    # touching sheets (no dark gap) with different labels: the label evidence decides under "either"
    V, lab = _slabs([(22, 31, 1, 1), (31, 40, 1, 2)])
    m = equivariance_switch(up(8.0), ref, 12.0, volume=V, label_volume=lab)
    assert m["switches"] == m["scored"] and equivariance_switch(up(8.0), ref, 12.0, volume=V, label_volume=lab,
                                                                confirm="ct")["switches"] == 0
    lv, d = sheet_crossing([[10, 10, 26.0]], [[10, 10, 40.0]], V, lab)
    assert lv[0] == 1 and d[0] > 0.9
    lv, d = sheet_crossing([[10, 10, 26.0]], [[10, 10, 40.0]])
    assert lv[0] == -1 and np.isnan(d[0])


def test_equivariance_switch_labels_and_streak_reference():
    ref = _plane()
    lt, lr = np.ones(ref.shape, int), np.ones(ref.shape, int)
    lt[10:20, 10:20] = 2  # same geometry, different human sheet at the match
    m = equivariance_switch(ref, ref, 12.0, lt, lr)
    assert m["switch_label"] == m["switches"] == 100 and m["scored"] == 28 * 38
    streak = np.full(ref.points.shape, np.nan)
    streak[15] = ref.points[15]  # reference is a 1-cell streak: no mesh, but present on this sheet
    flap = ref.points.copy() + [0, 0, 12.0]
    flap[:, :20] = np.nan  # ... and a flap one sheet away under the left half
    both = np.concatenate([streak, flap], 0)
    m = equivariance_switch(ref, Surface(both, ref.scale), 12.0)
    assert m["matched_by_vertex"] > 0 and not m["switch_map"][15].any()


def test_grid_rings_replays_growth():
    s = _plane(21, 21, seed_rc=(10, 10))
    ring = grid_rings(s)
    r, c = np.mgrid[0:21, 0:21]
    assert np.array_equal(ring, np.maximum(np.abs(r - 10) - 1, 0) + np.maximum(np.abs(c - 10) - 1, 0))
    P = s.points.copy()
    P[13, 6:15] = np.nan  # a wall: growth goes around it, one 2-chain at a time
    ring = grid_rings(Surface(P, s.scale, s.meta))
    v = np.isfinite(P[..., 0])
    assert (ring[v] >= 0).all() and (ring[~v] == -1).all() and ring[14, 10] > 3
    for (i, j) in zip(*np.nonzero(ring > 0)):  # every grown cell had a 2-chain of older cells
        assert any(0 <= ring[i + a, j + b] < ring[i, j] and 0 <= ring[i + 2 * a, j + 2 * b] < ring[i, j]
                   for a, b in ((0, 1), (0, -1), (1, 0), (-1, 0)) if 0 <= i + 2 * a < 21 and 0 <= j + 2 * b < 21)


def test_labels_offsheet_and_wilson():
    lab = np.zeros((64, 64, 64), np.int32)
    lab[28:33, :, :32] = 1
    lab[28:33, :, 32:] = 2  # the plane (z = 30) runs from sheet 1 onto sheet 2 at x = 32
    s = _plane(15, 20, tilt=0.0, seed_rc=(7, 3))
    v = vertex_labels(s, lab)
    assert set(np.unique(v)) == {1, 2} and (v[:, :11] == 1).all() and (v[:, 11:] == 2).all()
    a = label_agreement(s, lab, (0, 0, 0), 1)
    assert a["scored"] == 300 and abs(a["other_sheet"] - 0.45) < 1e-9
    assert offsheet_vs_labels(s, lab, 1, max_ring=6)["off_sheet"] == 0.0
    assert offsheet_vs_labels(s, lab, 1, max_ring=99)["off_sheet"] == a["other_sheet"]
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 1e-3
    lo, hi = wilson_ci(5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3
