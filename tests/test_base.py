import meshio
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg

from feat import base
from feat import boundary as bc
from feat import vector


def test_compute_E_material():
    element_type = "triangle"
    mesh_path = "tests/data/msh/test.msh"
    load_condition = "plane stress"  # "plane stress" or "plane strain"
    steel = base.Material("steel", 3e7, 0.25, load_condition)

    mesh = meshio.read(mesh_path)
    num_elements = mesh.cells_dict[element_type].shape[0]
    material_map = mesh.cell_data_dict["gmsh:physical"][element_type] - 1  # element-material map
    E_material = base.compute_E_material(num_elements, material_map, mesh.field_data, steel)

    E_steel = np.array([
        (3.2e7, 8e6, 0.0),
        (8e6, 3.2e7, 0.0),
        (0.0, 0.0, 1.2e7),
    ])
    # TODO add aluminum matrix testing
    np.testing.assert_allclose(E_steel, E_material[0])


def test_stiffness_matrix():
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
    E_material = base.compute_E_material(num_elements, material_map, mesh.field_data, steel)

    k_0 = base.stiffness_matrix(0, elements, nodal_coord, material_map, E_material, thickness, element_type)
    k_1 = base.stiffness_matrix(1, elements, nodal_coord, material_map, E_material, thickness, element_type)

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


def test_fem():
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
    
    E_material = base.compute_E_material(num_elements, material_map, mesh.field_data, steel)
    K = np.zeros((num_nodes * 2, num_nodes * 2))
    R = np.zeros(num_nodes * 2)
    K = base.assembly(K, num_elements, elements, nodal_coord, material_map, E_material, thickness, element_type)

    K, R = bc.apply_dirichlet(K, R, left_side, br_corner)
    R = bc.apply_neumann(R, tr_corner)

    D = np.linalg.solve(K, R)

    D_true = np.array([
        0.0,  0.0,
        1.90773874e-05,  0.0,
        8.73032981e-06, -7.41539125e-05,
        0.0,  0.0
    ])
    np.testing.assert_allclose(D_true, D)


def test_sparse_fem():
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
    
    E_material = base.compute_E_material(num_elements, material_map, mesh.field_data, steel)
    K = sparse.csc_matrix((2 * num_nodes, 2 * num_nodes))
    R = np.zeros(num_nodes * 2)
    # for e in range(num_elements):
    #     K = base.sparse_assembly(K, e, mesh, E_material, thickness, element_type, integration_points)
    K = base.sp_assembly(K, num_elements, num_nodes, elements, nodal_coord, material_map, E_material, thickness, element_type)

    print(K)
    K, R = bc.sp_apply_dirichlet(num_nodes, K, R, left_side, br_corner)
    R = bc.apply_neumann(R, tr_corner)

    D = linalg.spsolve(K, R)

    D_true = np.array([
        0.0,  0.0,
        1.90773874e-05,  0.0,
        8.73032981e-06, -7.41539125e-05,
        0.0,  0.0
    ])
    np.testing.assert_almost_equal(D_true, D)
