import meshio
import numpy as np
import pytest

from feat import base, vect
from feat.base import DirichletBC, NeumannBC


def test_compute_E_array():
    mesh_path = "tests/data/msh/test_mat.msh"
    load_condition = "plane stress"  # "plane stress" or "plane strain"
    steel = base.Material(1, 3e7, 0.25, load_condition)
    aluminum = base.Material(2, 1e7, 0.35, load_condition)
    
    mesh = meshio.read(mesh_path)

    E_array = vect.compute_E_array(mesh, steel, aluminum)
    E_array_true = np.array([
        (32000000., 8000000., 0., 32000000., 0., 12000000.),
        (11396011.3960114, 3988603.98860399, 0., 11396011.3960114, 0., 3703703.7037037),
    ])
    np.testing.assert_allclose(E_array_true, E_array)


def test_compute_K_entry():
    mesh_path = "tests/data/msh/test.msh"
    
    element_type = "T3"
    integration_points = 1
    load_condition = "plane stress"  # "plane stress" or "plane strain"
    thickness = 0.5
    steel = base.Material(1, 3e7, 0.25, load_condition)

    mesh = meshio.read(mesh_path)
    elements_num = mesh.cells["triangle"].shape[0]
    nodes = mesh.points.shape[0]
    elements = mesh.cells["triangle"]  # elements mapping, n-th row: nodes in n-th element
    coord = mesh.points[:,:2]  # x, y coordinates
    E_array = vect.compute_E_array(mesh, steel)

    row_0, col_0 = np.unravel_index(0, (6,6))
    k_0 = vect.compute_K_entry(row_0, col_0, coord, elements, E_array, thickness)
    
    row_35, col_35 = np.unravel_index(35, (6,6))
    k_35 = vect.compute_K_entry(row_35, col_35, coord, elements, E_array, thickness)

    k_0_true = np.array([5333333.33333333, 4500000.])
    k_35_true = np.array([12000000., 14000000.])

    np.testing.assert_allclose(k_0_true, k_0)
    np.testing.assert_allclose(k_35_true, k_35)


def test_compute_global_dof():
    mesh_path = "tests/data/msh/test.msh"
    mesh = meshio.read(mesh_path)

    row, col = np.unravel_index(8, (6,6))
    row_ind, col_ind = vect.compute_global_dof(mesh, row, col)

    row_ind_true = np.array([1, 1])
    col_ind_true = np.array([2, 4])

    np.testing.assert_allclose(row_ind_true, row_ind)
    np.testing.assert_allclose(col_ind_true, col_ind)

    del row; del col;
    del row_ind; del col_ind;
    del row_ind_true; del col_ind_true;
    
    row, col = np.unravel_index(29, (6,6))
    row_ind, col_ind = vect.compute_global_dof(mesh, row, col)

    row_ind_true = np.array([4, 6])
    col_ind_true = np.array([5, 7])

    np.testing.assert_allclose(row_ind_true, row_ind)
    np.testing.assert_allclose(col_ind_true, col_ind)


def test_vect_assembly():
    mesh_path = "tests/data/msh/test.msh"
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
    np.testing.assert_allclose(K_true, K.toarray())

    K, R = vect.apply_dirichlet(nodes, K, R, left_side, br_corner)
    R = base.apply_neumann(R, tr_corner)
 
    K_true_bc = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-5.33333333e+06, 3.00000000e+06, 9.83333333e+06, -5.00000000e+06, -4.50000000e+06, 2.00000000e+06, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -5.00000000e+06,-4.50000000e+06, 3.00000000e+06, 9.83333333e+06, 0.0, -5.33333333e+06, 2.00000000e+06],
        [-5.00000000e+06, 0.0, 2.00000000e+06, -1.20000000e+07, 0.0, 1.40000000e+07, 3.00000000e+06, -2.00000000e+06],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    np.testing.assert_allclose(K_true_bc, K.toarray())