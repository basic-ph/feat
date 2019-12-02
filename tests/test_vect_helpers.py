
import numpy as np
import pytest

from feat.boundary import DirichletBC, NeumannBC, dirichlet_dof
from feat.helpers import compute_E_matrices, gauss_quadrature
from feat.vect_helpers import (vect_assembly, vect_compute_E,
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

    np.testing.assert_allclose(k_0_true, k_0)
    np.testing.assert_allclose(k_35_true, k_35)

# @pytest.mark.skip
def test_vect_compute_global_dof(setup_mesh):
    mesh = setup_mesh("gmsh/msh/test.msh")

    row, col = np.unravel_index(8, (6,6))
    row_ind, col_ind = vect_compute_global_dof(mesh, row, col)

    row_ind_true = np.array([1, 1])
    col_ind_true = np.array([2, 4])

    np.testing.assert_allclose(row_ind_true, row_ind)
    np.testing.assert_allclose(col_ind_true, col_ind)

    del row; del col;
    del row_ind; del col_ind;
    del row_ind_true; del col_ind_true;
    
    row, col = np.unravel_index(29, (6,6))
    row_ind, col_ind = vect_compute_global_dof(mesh, row, col)

    row_ind_true = np.array([4, 6])
    col_ind_true = np.array([5, 7])

    np.testing.assert_allclose(row_ind_true, row_ind)
    np.testing.assert_allclose(col_ind_true, col_ind)


# @pytest.mark.skip
def test_vect_assembly(setup_data, setup_mesh):
    data = setup_data("data/test.json")
    mesh = setup_mesh("gmsh/msh/test.msh")
    elements_num = mesh.cells["triangle"].shape[0]

    left_side = DirichletBC("left side", data, mesh)
    br_corner = DirichletBC("bottom right corner", data, mesh)
    tr_corner = NeumannBC("top right corner", data, mesh)

    K, K_stored, R = vect_assembly(data, mesh, left_side, br_corner)

    K_true = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 9.83333333e+06, 0.0, -4.50000000e+06, 2.00000000e+06, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0,-4.50000000e+06, 0.0, 9.83333333e+06, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.00000000e+06, 0.0, 0.0, 1.40000000e+07, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    K_stored_true = np.array([
        [9833333.33333333, 0., -5333333.33333333, 2000000., 0., -5000000., -4500000., 3000000.],
        [0., 14000000., 3000000., -2000000., -5000000., 0.,  2000000.,-12000000.],
        [-5333333.33333333, 3000000., 9833333.33333333, -5000000., -4500000.,  2000000., 0., 0.],
        [2000000., -2000000., -5000000., 14000000., 3000000., -12000000., 0., 0.],
        [0., -5000000., -4500000., 3000000., 9833333.33333333, 0., -5333333.33333333, 2000000.],
        [-5000000., 0., 2000000., -12000000., 0., 14000000., 3000000., -2000000.],
        [-4500000., 2000000., 0., 0., -5333333.33333333, 3000000., 9833333.33333333, -5000000.],
        [3000000., -12000000., 0., 0., 2000000., -2000000., -5000000., 14000000.],
    ])

    R_true = np.zeros(8)
    np.set_printoptions(linewidth=200)
    # print(K_true)
    # print()
    # print(K.toarray())

    for i in range(8):
        for j in range(8):
            np.testing.assert_allclose(K_true[i,j], K[i,j])
            np.testing.assert_allclose(K_stored_true[i,j], K_stored[i,j])
    np.testing.assert_allclose(R_true, R)
