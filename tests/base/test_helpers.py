import meshio
import numpy as np
import pytest

from feat import base


def test_compute_E_array():
    mesh_path = "tests/data/msh/test.msh"
    load_condition = "plane stress"  # "plane stress" or "plane strain"
    steel = base.Material(1, 3e7, 0.25, load_condition)

    mesh = meshio.read(mesh_path)    
    E_array = base.compute_E_array(mesh, steel)

    E_steel = np.array([
        (3.2e7, 8e6, 0.0),
        (8e6, 3.2e7, 0.0),
        (0.0, 0.0, 1.2e7),
    ])
    # TODO add aluminum matrix testing
    np.testing.assert_allclose(E_steel, E_array[0])


def test_stiffness_matrix():
    mesh_path = "tests/data/msh/test.msh"

    element_type = "T3"
    integration_points = 1
    load_condition = "plane stress"  # "plane stress" or "plane strain"
    thickness = 0.5
    steel = base.Material(1, 3e7, 0.25, load_condition)

    mesh = meshio.read(mesh_path)
    elements_num = mesh.cells["triangle"].shape[0]
    nodes = mesh.points.shape[0]
    E_array = base.compute_E_array(mesh, steel)

    k_0 = base.stiffness_matrix(0, mesh, E_array, thickness, element_type, integration_points)
    k_1 = base.stiffness_matrix(1, mesh, E_array, thickness, element_type, integration_points)

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
