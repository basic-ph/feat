
import numpy as np
import pytest

from feat.helpers import compute_E_matrices, gauss_quadrature
from feat.vect_helpers import (assembly_opt_v1, vect_compute_E,
                               vect_compute_global_dof, vect_compute_K_entry,
                               vect_stiffness_matrix)


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


# @pytest.mark.skip
def test_vect_stiffness_matrix(setup_data, setup_mesh):
    data = setup_data("data/test.json")
    mesh = setup_mesh("gmsh/msh/test.msh")
    elements_num = mesh.cells["triangle"].shape[0]

    E_array = vect_compute_E(data, mesh, elements_num)

    K_array = vect_stiffness_matrix(data, mesh, E_array)

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


# @pytest.mark.skip
def test_vect_compute_K_entry(setup_data, setup_mesh):
    data = setup_data("data/test.json")
    mesh = setup_mesh("gmsh/msh/test.msh")
    
    t = data["thickness"]
    elements_num = mesh.cells["triangle"].shape[0]
    e = mesh.cells["triangle"]  # elements mapping, n-th row: nodes in n-th element
    c = mesh.points[:,:2]  # x, y coordinates
    E_array = vect_compute_E(data, mesh, elements_num)


    k_0 = vect_compute_K_entry(0, c, e, E_array, t)
    k_35 = vect_compute_K_entry(35, c, e, E_array, t)

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
