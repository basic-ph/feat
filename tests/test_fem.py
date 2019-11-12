import numpy as np

from feat.boundary import DirichletBC, NeumannBC, dirichlet_dof
from feat.helpers import (assembly, compute_E_matrices, gauss_quadrature,
                          stiffness_matrix)
from feat.post_proc import compute_modulus


def test_base(setup_data, setup_mesh):
    data = setup_data("data/test.json")
    weights, locations = gauss_quadrature(data)
    mesh = setup_mesh("gmsh/msh/test.msh")
    elements = mesh.cells["triangle"].shape[0]
    nodes = mesh.points.shape[0]

    E_matrices = compute_E_matrices(data, mesh)
    K = np.zeros((nodes * 2, nodes * 2))
    R = np.zeros(nodes * 2)

    for e in range(elements):  # number of elements
        K = assembly(e, data, mesh, E_matrices, K)

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
