import numpy as np
import pytest

from feat.helpers import compute_E_matrices, gauss_quadrature, stiffness_matrix


def test_compute_E_matrices(setup_data, setup_mesh):
    data = setup_data(r"data/test_mat.json")
    mesh = setup_mesh(r"gmsh/msh/test_mat.msh")

    E_matrices = compute_E_matrices(data, mesh)
    E_steel = np.array([
        (3.2e7, 8e6, 0.0),
        (8e6, 3.2e7, 0.0),
        (0.0, 0.0, 1.2e7),
    ])
    # TODO add aluminum matrix testing
    np.testing.assert_allclose(E_steel, E_matrices[0])


def test_stiffness_matrix(setup_data, setup_mesh):
    data = setup_data("data/test.json")
    weights, locations = gauss_quadrature(data)
    mesh = setup_mesh("gmsh/msh/test.msh")
    elements = mesh.cells["triangle"].shape[0]
    nodes = mesh.points.shape[0]
    E_matrices = compute_E_matrices(data, mesh)

    k_0 = stiffness_matrix(0, data, mesh, E_matrices)
    k_1 = stiffness_matrix(1, data, mesh, E_matrices)

    k_0_true = np.array([
        (5333333.33333333, 0.0, -5333333.33333333, 2000000., 0., -2000000.),
        (0., 2000000., 3000000., -2000000., -3000000., 0.),
        (-5333333.33333333, 3000000., 9833333.33333333, -5000000., -4500000., 2000000.),
        (2000000.,-2000000., -5000000., 14000000., 3000000., -12000000.),
        (0., -3000000., -4500000., 3000000., 4500000., 0.),
        (-2000000., 0., 2000000., -12000000., 0., 12000000.),
    ])
    k_1_true = np.array([
        (4500000., 0., 0., -3000000., -4500000., 3000000.),
        (0., 12000000., -2000000., 0., 2000000., -12000000.),
        (0., -2000000., 5333333.33333333, 0., -5333333.33333333, 2000000.),
        (-3000000., 0., 0., 2000000., 3000000., -2000000.),
        (-4500000., 2000000., -5333333.33333333, 3000000., 9833333.33333333, -5000000.),
        (3000000., -12000000., 2000000., -2000000., -5000000., 14000000.),
    ])

    np.testing.assert_allclose(k_0_true, k_0)
    np.testing.assert_allclose(k_1_true, k_1)
