
import numpy as np
import pytest

from feat.helpers import compute_E_matrices, gauss_quadrature
from feat.vect_helpers import (assembly_opt_v1, vect_assembly, vect_compute_E,
                               vect_compute_global_dof, vect_compute_K_entry)


def test_compute_E_vect(setup_data, setup_mesh):
    data = setup_data("data/test_mat.json")
    mesh = setup_mesh("gmsh/msh/test_mat.msh")
    elements_num = mesh.cells["triangle"].shape[0]

    E_array = vect_compute_E(data, mesh, elements_num)
    print(E_array)
    E_array_true = np.array([
        (32000000., 8000000., 0., 32000000., 0., 12000000.),
        (11396011.3960114, 3988603.98860399, 0., 11396011.3960114, 0., 3703703.7037037),
    ])
    np.testing.assert_allclose(E_array_true, E_array)


def test_vect_compute_K_entry(setup_data, setup_mesh):
    data = setup_data("data/test.json")
    mesh = setup_mesh("gmsh/msh/test.msh")
    
    t = data["thickness"]
    elements_num = mesh.cells["triangle"].shape[0]
    e = mesh.cells["triangle"]  # elements mapping, n-th row: nodes in n-th element
    c = mesh.points[:,:2]  # x, y coordinates
    E_array = vect_compute_E(data, mesh, elements_num)

    row_0, col_0 = np.unravel_index(0, (6,6))
    k_0 = vect_compute_K_entry(row_0, col_0, c, e, E_array, t)
    
    row_35, col_35 = np.unravel_index(35, (6,6))
    k_35 = vect_compute_K_entry(row_35, col_35, c, e, E_array, t)

    k_0_true = np.array([5333333.33333333, 4500000.])
    k_35_true = np.array([12000000., 14000000.])

    print(k_0)
    print()
    print(k_0_true)
    np.testing.assert_allclose(k_0_true, k_0)
    np.testing.assert_allclose(k_35_true, k_35)


def test_vect_compute_global_dof(setup_mesh):
    mesh = setup_mesh("gmsh/msh/test.msh")

    element_dof = vect_compute_global_dof(mesh)

    element_dof_true = np.array([
        (0, 1, 2, 3, 4, 5),
        (0, 1, 4, 5, 6 ,7),
    ])
    np.testing.assert_allclose(element_dof_true, element_dof)


def test_vect_assembly(setup_data, setup_mesh):
    data = setup_data("data/test.json")
    mesh = setup_mesh("gmsh/msh/test.msh")
    elements_num = mesh.cells["triangle"].shape[0]

    E_array = vect_compute_E(data, mesh, elements_num)

    K_array, I_array, J_array = vect_assembly(data, mesh, E_array)

    k_0_true = np.array([
        5333333.33333333, 0.0, -5333333.33333333, 2000000., 0., -2000000.,
        0., 2000000., 3000000., -2000000., -3000000., 0.,
        -5333333.33333333, 3000000., 9833333.33333333, -5000000., -4500000., 2000000.,
        2000000.,-2000000., -5000000., 14000000., 3000000., -12000000.,
        0., -3000000., -4500000., 3000000., 4500000., 0.,
        -2000000., 0., 2000000., -12000000., 0., 12000000.,
    ])
    k_1_true = np.array([
        4500000., 0., 0., -3000000., -4500000., 3000000.,
        0., 12000000., -2000000., 0., 2000000., -12000000.,
        0., -2000000., 5333333.33333333, 0., -5333333.33333333, 2000000.,
        -3000000., 0., 0., 2000000., 3000000., -2000000.,
        -4500000., 2000000., -5333333.33333333, 3000000., 9833333.33333333, -5000000.,
        3000000., -12000000., 2000000., -2000000., -5000000., 14000000.,
    ])

    np.testing.assert_allclose(k_0_true, K_array[:,0])  # comparing with first col of K_array
    np.testing.assert_allclose(k_1_true, K_array[:,1])

    I_array_true = np.array([
        [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.],
        [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
        [2., 4.], [2., 4.], [2., 4.], [2., 4.], [2., 4.], [2., 4.],
        [3., 5.], [3., 5.], [3., 5.], [3., 5.], [3., 5.], [3., 5.],
        [4., 6.], [4., 6.], [4., 6.], [4., 6.], [4., 6.], [4., 6.],
        [5., 7.], [5., 7.], [5., 7.], [5., 7.], [5., 7.], [5., 7.],
    ])
    J_array_true = np.array([
        [0., 0.], [1., 1.], [2., 4.], [3., 5.], [4., 6.], [5., 7.],
        [0., 0.], [1., 1.], [2., 4.], [3., 5.], [4., 6.], [5., 7.],
        [0., 0.], [1., 1.], [2., 4.], [3., 5.], [4., 6.], [5., 7.],
        [0., 0.], [1., 1.], [2., 4.], [3., 5.], [4., 6.], [5., 7.],
        [0., 0.], [1., 1.], [2., 4.], [3., 5.], [4., 6.], [5., 7.],
        [0., 0.], [1., 1.], [2., 4.], [3., 5.], [4., 6.], [5., 7.],
    ])

    np.testing.assert_allclose(I_array_true, I_array)
    np.testing.assert_allclose(J_array_true, J_array)
