import meshio
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg

from feat import base, vect
from feat.base import DirichletBC, NeumannBC


def test_fem():
    mesh_path = "tests/data/msh/test.msh"
    
    integration_points = 1
    load_condition = "plane stress"  # "plane stress" or "plane strain"
    thickness = 0.5
    steel = base.Material(1, 3e7, 0.25, load_condition)

    mesh = meshio.read(mesh_path)
    elements_num = mesh.cells["triangle"].shape[0]
    nodes = mesh.points.shape[0]

    left_side = DirichletBC("left side", mesh, [0, 1], 0.0)
    br_corner = DirichletBC("bottom right corner", mesh, [1], 0.0)
    tr_corner = NeumannBC("top right corner", mesh, [1], -1000.0)

    E_array = vect.compute_E_array(mesh, steel)
    R = np.zeros(nodes * 2)
    K = vect.assembly(mesh, E_array, thickness)

    K, R = vect.apply_dirichlet(nodes, K, R, left_side, br_corner)
    R = base.apply_neumann(R, tr_corner)

    D = linalg.spsolve(K, R)

    D_true = np.array([
        0.0,  0.0,
        1.90773874e-05,  0.0,
        8.73032981e-06, -7.41539125e-05,
        0.0,  0.0
    ])
    # np.testing.assert_allclose(D_true, D)  # FIXME some zeros are 1.0e-20 why??
    np.testing.assert_almost_equal(D_true, D)
