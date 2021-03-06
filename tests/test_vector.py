import meshio
import numpy as np
import pytest
from scipy.sparse import linalg

from feat import base
from feat import boundary as bc
from feat import vector


def test_compute_E_array():
    element_type = "triangle"
    mesh_path = "tests/data/msh/test_mat.msh"
    load_condition = "plane stress"  # "plane stress" or "plane strain"
    steel = base.Material("steel", 3e7, 0.25, load_condition)
    aluminum = base.Material("aluminum", 1e7, 0.35, load_condition)
    
    mesh = meshio.read(mesh_path)
    elements = mesh.cells_dict[element_type]
    nodal_coord = mesh.points[:,:2]
    num_elements = elements.shape[0]
    material_map = mesh.cell_data_dict["gmsh:physical"][element_type] - 1  # element-material map

    E_array = vector.compute_E_array(num_elements, material_map, mesh.field_data, steel, aluminum)
    E_array_true = np.array([
        (32000000., 8000000., 0., 32000000., 0., 12000000.),
        (11396011.3960114, 3988603.98860399, 0., 11396011.3960114, 0., 3703703.7037037),
    ])
    np.testing.assert_allclose(E_array_true, E_array)

# @pytest.mark.skip()
def test_compute_K_entry():
    mesh_path = "tests/data/msh/test.msh"
    
    element_type = "triangle"
    integration_points = 1
    load_condition = "plane stress"  # "plane stress" or "plane strain"
    thickness = 0.5
    steel = base.Material("steel", 3e7, 0.25, load_condition)

    mesh = meshio.read(mesh_path)
    elements = mesh.cells_dict[element_type]
    nodal_coord = mesh.points[:,:2]
    num_elements = elements.shape[0]
    num_nodes = nodal_coord.shape[0]
    material_map = mesh.cell_data_dict["gmsh:physical"][element_type] - 1  # element-material map
    E_array = vector.compute_E_array(num_elements, material_map, mesh.field_data, steel)

    X = lambda c, e, i, j: c[e[:,i]][:,0] - c[e[:,j]][:,0]
    Y = lambda c, e, i, j: c[e[:,i]][:,1] - c[e[:,j]][:,1]
    c = nodal_coord
    e = elements
    J = X(c,e,1,0) * Y(c,e,2,0) - X(c,e,2,0) * Y(c,e,1,0)
    b = np.array([
        [Y(c,e,1,2), Y(c,e,2,0), Y(c,e,0,1)],
        [X(c,e,2,1), X(c,e,0,2), X(c,e,1,0)],
    ])

    row_0, col_0 = np.unravel_index(0, (6,6))
    k_0 = vector.compute_K_entry(row_0, col_0, nodal_coord, elements, b, J, E_array, thickness)
    
    row_35, col_35 = np.unravel_index(35, (6,6))
    k_35 = vector.compute_K_entry(row_35, col_35, nodal_coord, elements, b, J, E_array, thickness)

    k_0_true = np.array([5333333.33333333, 4500000.])
    k_35_true = np.array([12000000., 14000000.])

    np.testing.assert_allclose(k_0_true, k_0)
    np.testing.assert_allclose(k_35_true, k_35)


def test_compute_global_dof():
    element_type = "triangle"
    mesh_path = "tests/data/msh/test.msh"
    mesh = meshio.read(mesh_path)
    elements = mesh.cells_dict[element_type]
    num_elements = elements.shape[0]

    row, col = np.unravel_index(8, (6,6))
    row_ind, col_ind = vector.compute_global_dof(num_elements, elements, row, col)

    row_ind_true = np.array([1, 1])
    col_ind_true = np.array([2, 4])

    np.testing.assert_allclose(row_ind_true, row_ind)
    np.testing.assert_allclose(col_ind_true, col_ind)

    del row; del col;
    del row_ind; del col_ind;
    del row_ind_true; del col_ind_true;
    
    row, col = np.unravel_index(29, (6,6))
    row_ind, col_ind = vector.compute_global_dof(num_elements, elements, row, col)

    row_ind_true = np.array([4, 6])
    col_ind_true = np.array([5, 7])

    np.testing.assert_allclose(row_ind_true, row_ind)
    np.testing.assert_allclose(col_ind_true, col_ind)


def test_vect_assembly():
    mesh_path = "tests/data/msh/test.msh"
    element_type = "triangle"
    load_condition = "plane stress"  # "plane stress" or "plane strain"
    thickness = 0.5
    steel = base.Material("steel", 3e7, 0.25, load_condition)

    mesh = meshio.read(mesh_path)
    elements = mesh.cells_dict[element_type]
    nodal_coord = mesh.points[:,:2]
    num_elements = elements.shape[0]
    num_nodes = nodal_coord.shape[0]
    material_map = mesh.cell_data_dict["gmsh:physical"][element_type] - 1  # element-material map

    left_side = bc.DirichletBC("left side", mesh, [0, 1], 0.0)
    br_corner = bc.DirichletBC("bottom right corner", mesh, [1], 0.0)
    tr_corner = bc.NeumannBC("top right corner", mesh, [1], -1000.0)

    E_array = vector.compute_E_array(num_elements, material_map, mesh.field_data, steel)
    R = np.zeros(num_nodes * 2)
    K = vector.assembly(num_elements, num_nodes, elements, nodal_coord, E_array, thickness)
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

    K, R = bc.sp_apply_dirichlet(num_nodes, K, R, left_side, br_corner)
    R = bc.apply_neumann(R, tr_corner)
 
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


def test_fem():
    mesh_path = "tests/data/msh/test.msh"
    
    element_type = "triangle"
    integration_points = 1
    load_condition = "plane stress"  # "plane stress" or "plane strain"
    thickness = 0.5
    steel = base.Material("steel", 3e7, 0.25, load_condition)

    mesh = meshio.read(mesh_path)
    elements = mesh.cells_dict[element_type]
    nodal_coord = mesh.points[:,:2]
    num_elements = elements.shape[0]
    num_nodes = nodal_coord.shape[0]
    material_map = mesh.cell_data_dict["gmsh:physical"][element_type] - 1  # element-material map

    left_side = bc.DirichletBC("left side", mesh, [0, 1], 0.0)
    br_corner = bc.DirichletBC("bottom right corner", mesh, [1], 0.0)
    tr_corner = bc.NeumannBC("top right corner", mesh, [1], -1000.0)

    E_array = vector.compute_E_array(num_elements, material_map, mesh.field_data, steel)
    R = np.zeros(num_nodes * 2)
    K = vector.assembly(num_elements, num_nodes, elements, nodal_coord, E_array, thickness)

    K, R = bc.sp_apply_dirichlet(num_nodes, K, R, left_side, br_corner)
    R = bc.apply_neumann(R, tr_corner)

    D = linalg.spsolve(K, R)

    D_true = np.array([
        0.0,  0.0,
        1.90773874e-05,  0.0,
        8.73032981e-06, -7.41539125e-05,
        0.0,  0.0
    ])
    # np.testing.assert_allclose(D_true, D)  # FIXME some zeros are 1.0e-20 why??
    np.testing.assert_almost_equal(D_true, D)
