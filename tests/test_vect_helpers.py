
import numpy as np
import pytest

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


def test_vect_compute_global_dof(setup_mesh):
    mesh = setup_mesh("gmsh/msh/test.msh")

    row, col = np.unravel_index(8, (6,6))
    I_indices, J_indices = vect_compute_global_dof(mesh, row, col)

    I_indices_true = np.array([1, 1])
    J_indices_true = np.array([2, 4])

    np.testing.assert_allclose(I_indices_true, I_indices)
    np.testing.assert_allclose(J_indices_true, J_indices)

    del row; del col;
    del I_indices; del J_indices;
    del I_indices_true; del J_indices_true;
    
    row, col = np.unravel_index(29, (6,6))
    I_indices, J_indices = vect_compute_global_dof(mesh, row, col)

    I_indices_true = np.array([4, 6])
    J_indices_true = np.array([5, 7])

    np.testing.assert_allclose(I_indices_true, I_indices)
    np.testing.assert_allclose(J_indices_true, J_indices)


# @pytest.mark.skip
def test_vect_assembly(setup_data, setup_mesh):
    data = setup_data("data/test.json")
    mesh = setup_mesh("gmsh/msh/test.msh")
    elements_num = mesh.cells["triangle"].shape[0]

    K = vect_assembly(data, mesh)  # csr format

    K_true = np.array([
        [9833333.33333333, 0., -5333333.33333333, 2000000., 0., -5000000., -4500000., 3000000.],
        [0., 14000000., 3000000., -2000000., -5000000., 0.,  2000000.,-12000000.],
        [-5333333.33333333, 3000000., 9833333.33333333, -5000000., -4500000.,  2000000., 0., 0.],
        [2000000., -2000000., -5000000., 14000000., 3000000., -12000000., 0., 0.],
        [0., -5000000., -4500000., 3000000., 9833333.33333333, 0., -5333333.33333333, 2000000.],
        [-5000000., 0., 2000000., -12000000., 0., 14000000., 3000000., -2000000.],
        [-4500000., 2000000., 0., 0., -5333333.33333333, 3000000., 9833333.33333333, -5000000.],
        [3000000., -12000000., 0., 0., 2000000., -2000000., -5000000., 14000000.],
    ])

    for i in range(8):
        for j in range(8):
            np.testing.assert_allclose(K_true[i,j], K[i,j])
