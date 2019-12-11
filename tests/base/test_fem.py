import meshio
import numpy as np
import pytest

from feat import base
from feat.base import DirichletBC, NeumannBC


def test_fem():
    mesh_path = "tests/data/msh/test.msh"

    element_type = "T3"
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
    
    E_array = base.compute_E_array(mesh, steel)
    K = np.zeros((nodes * 2, nodes * 2))
    R = np.zeros(nodes * 2)
    for e in range(elements_num):
        base.assembly(K, e, mesh, E_array, thickness, element_type, integration_points)

    K, R = base.apply_dirichlet(K, R, left_side, br_corner)
    R = base.apply_neumann(R, tr_corner)

    D = np.linalg.solve(K, R)

    D_true = np.array([
        0.0,  0.0,
        1.90773874e-05,  0.0,
        8.73032981e-06, -7.41539125e-05,
        0.0,  0.0
    ])
    np.testing.assert_allclose(D_true, D)
