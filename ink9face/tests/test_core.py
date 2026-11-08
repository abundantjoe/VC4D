"""Unit tests (synthetic data only).  python -m pytest   (or python tests/test_core.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ink9face import face, labels, metrics, tifxyz  # noqa: E402


def plane(H=8, W=9):
    r, c = np.meshgrid(np.arange(H, dtype=float), np.arange(W, dtype=float), indexing="ij")
    return np.stack([c * 20, r * 20, np.full_like(r, 100.0)], -1)   # x along columns, y along rows, z constant


def test_vc_normal_sign():
    # VC3D grid_normal: xv (along cols) = +x, yv (along rows) = +y  ->  n = xv x yv = +z
    n = tifxyz.vc_normals(plane())
    assert np.allclose(n[..., 2], 1.0)


def test_vc_normal_holes():
    P = plane(); P[3, 4] = np.nan
    n = tifxyz.vc_normals(P)
    assert np.isnan(n[3, 4]).all() and np.allclose(n[3, 3, 2], 1.0) and np.allclose(n[2, 4, 2], 1.0)


def test_orient():
    st = np.arange(5)[:, None, None] * np.ones((5, 2, 2))
    assert face.orient(st, "anti_normal")[0, 0, 0] == 4 and face.orient(st, "native")[0, 0, 0] == 0


def test_rule_identity():
    gr, gc = labels.rule(np.array([0.0, 10.0]), np.array([0.0, 10.0]), (11, 11), (11, 11))
    assert np.allclose(gr, [0, 10]) and np.allclose(gc, [0, 10])


def test_affine_fit():
    rng = np.random.default_rng(0); A = rng.normal(size=(100, 3))
    T = np.c_[np.diag([4.2, 4.2, 4.3]), [1, 2, 3]]
    assert np.allclose(labels.fit_affine(A, labels.apply_affine(T, A)), T)


def test_auc_perfect():
    p = np.r_[np.zeros(100), np.ones(100)]; y = p > 0.5
    assert metrics.auc(p, y) == 1.0


def test_mesh_paired():
    r = metrics.mesh_paired([0.1, 0.1, 0.2, -0.05], ["a", "a", "b", "c"])
    assert r["n_meshes"] == 3 and r["better"] == 2


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
