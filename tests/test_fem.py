import numpy as np

from feat.boundary import DirichletBC, NeumannBC, dirichlet_dof
from feat.helpers import (assembly, compute_E_matrices, gauss_quadrature,
                          stiffness_matrix)
from feat.post_proc import compute_modulus


def test_script(setup_data, setup_mesh):
    data = setup_data("data/test.json")
    element_type = data["element type"]
    ip_number = data["integration points"]
    thickness = data["thickness"]
    post = data["post-processing"]
    weights, locations = gauss_quadrature(data)
    mesh = setup_mesh("gmsh/msh/test.msh")
    nodal_coordinates = mesh.points[:,:2]  # slice is used to remove 3rd coordinate
    nodes = mesh.points.shape[0]
    dof = nodes * 2
    connectivity_table = mesh.cells["triangle"]
    elements = connectivity_table.shape[0]
    element_material_map = mesh.cell_data["triangle"]["gmsh:physical"]
    E_matrices = compute_E_matrices(data, mesh)
    K = np.zeros((dof, dof))
    R = np.zeros(dof)

    for e in range(elements):
            k = stiffness_matrix(
                    e,
                    data,
                    mesh,
                    nodal_coordinates,
                    connectivity_table,
                    element_material_map,
                    E_matrices
            )
            K = assembly(e, connectivity_table, k, K)

    left_side = DirichletBC("left side", data, mesh)
    br_corner = DirichletBC("bottom right corner", data, mesh)
    tr_corner = NeumannBC("top right corner", data, mesh)
    left_side.impose(K, R)
    br_corner.impose(K, R)
    tr_corner.impose(R)
    D = np.linalg.solve(K, R)
    D_true = np.array([
        0.0,  0.0,
        1.90773874e-05,  0.0,
        8.73032981e-06, -7.41539125e-05,
        0.0,  0.0
    ])
    np.testing.assert_allclose(D_true, D)
