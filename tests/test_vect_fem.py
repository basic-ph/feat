import numpy as np
from scipy import sparse

from feat.helpers import gauss_quadrature
from feat.vect_helpers import (vect_assembly, vect_compute_E,
                               vect_compute_global_dof, vect_compute_K_entry)


def test_vect_fem(setup_data, setup_mesh):
    data = setup_data("data/test.json")
    weights, locations = gauss_quadrature(data)
    mesh = setup_mesh("gmsh/msh/test.msh")
    elements_num = mesh.cells["triangle"].shape[0]
    nodes = mesh.points.shape[0]

    R = np.zeros(nodes * 2)
    K_array, I_array, J_array = vect_assembly(data, mesh)
    K = sparse.csc_matrix((np.ravel(K_array), (np.ravel(I_array), np.ravel(J_array))), shape=(2 * nodes, 2 * nodes))

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
            print(K[i,j])
            np.testing.assert_allclose(K_true[i,j], K[i,j])