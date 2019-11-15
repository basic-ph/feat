import sys

import numpy as np
import pytest

sys.path.insert(0, "/home/basic-ph/thesis/feat/feat/")
from feat.helpers import compute_E_matrices, gauss_quadrature
from feat.vector import assembly_opt_v1, compute_element_global_dof_vect


def test_assembly_opt_v1(setup_data, setup_mesh):
    data = setup_data("data/test.json")
    weights, locations = gauss_quadrature(data)
    mesh = setup_mesh("gmsh/msh/test.msh")
    elements = mesh.cells["triangle"].shape[0]
    nodes = mesh.points.shape[0]

    E_matrices = compute_E_matrices(data, mesh)
    K_flat = np.zeros(36 * elements)  # 36 is 6^2 (dofs^2)
    I = np.zeros(36 * elements, dtype=np.int32)  # the 2nd quantity is the number of elements
    J = np.zeros(36 * elements, dtype=np.int32)
    # testing only with element 1 (the 2nd)
    for e in range(elements):  # number of elements
        K_flat, I, J = assembly_opt_v1(e, data, mesh, E_matrices, K_flat, I, J)
    print(K_flat)

    I_true = np.array([
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 4, 5, 6, 7,
        0, 1, 4, 5, 6, 7,
        0, 1, 4, 5, 6, 7,
        0, 1, 4, 5, 6, 7,
        0, 1, 4, 5, 6, 7,
        0, 1, 4, 5, 6, 7,
    ], dtype=np.int32)
    J_true = np.array([
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5,
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1,
        4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7,
    ], dtype=np.int32)
    
    np.testing.assert_allclose(I_true, I)
    np.testing.assert_allclose(J_true, J)


def test_1(setup_mesh):
    mesh = setup_mesh("gmsh/msh/test.msh")

    element_dof = compute_element_global_dof_vect(mesh)

    assert False