
import sys

import numpy as np
from numpy.linalg import det, inv
from tools import gauss_quadrature

x = lambda a, i, j: a[i][0] - a[j][0]
y = lambda b, i, j: b[i][1] - b[j][1]


def stiffness_matrix(e, data, mesh, coordinates, connectivity, material_map, E_matrices):

    t = data["thickness"]
    element_nodes = connectivity[e]
    # print("nodes:\n", element_nodes)
    c = coordinates[element_nodes]  # element coordinates
    # print("coord:\n", c)

    element_material = material_map[e]
    E = E_matrices[element_material]["E"]
    # print("E:\n", E)

    # element/local stiffness matrix
    k = np.zeros((6, 6))  # for T6 --> 12 by 12

    weights, locations = gauss_quadrature(data)
    # print(weights)
    # print(locations)
    for p in range(weights.shape[0]):
        w = weights[p]
        loc = locations[p]  # this is a [x, y] array
        
        j = ( (c[1][0] - c[0][0]) * (c[2][1] - c[0][1])  # det of jacobian matrix
            - (c[2][0] - c[0][0]) * (c[1][1] - c[0][1])
        )
        # print(j)
        B = (1/j) * np.array([
            (y(c, 1, 2), 0, y(c, 2, 0), 0, y(c, 0, 1), 0),
            (0, x(c, 2, 1), 0, x(c, 0, 2), 0 , x(c, 1, 0)),
            (x(c, 2, 1), y(c, 1, 2), x(c, 0, 2), y(c, 2, 0), x(c, 1, 0), y(c, 0, 1))
        ])
        # print(B)
        k_integral = B.T @ E @ B * t * 0.5 * j * w
        k += k_integral

    return k


def assembly(e, connectivity, k, K):
    element_nodes = connectivity[e]
    element_dofs = np.zeros(6, dtype=np.int32)  # becomes 12 for T6
    for n in range(element_nodes.shape[0]):  # TODO check if applicable for BC
        element_dofs[n*2] = element_nodes[n] * 2
        element_dofs[n*2+1] = element_nodes[n] * 2 + 1
    # print(element_dofs)  

    for i in range(6):  # becomes 12 for T6
        I = element_dofs[i]
        for j in range(6):  # becomes 12 for T6
            J = element_dofs[j]
            K[I, J] += k[i, j]
    return K


def dirichlet_dof(*conditions):
    array_list = [c.global_dof for c in conditions]
    total_dof = np.concatenate(array_list)
    return total_dof
