import sys

import numpy as np
from numpy.linalg import det, inv


def compute_E_matrices(data, mesh):
    condition = data["load condition"]
    E_matrices = {}

    for key,value in data["materials"].items():
        # key is the material name
        # value is the dict with young's modulus and poisson's ratio
        physical_tag = mesh.field_data[key][0]
        
        poisson = value["poisson's ratio"]
        young = value["young's modulus"]

        if condition == "plane strain":
            coeff = young / ((1 + poisson) * (1 - 2 * poisson))
            matrix = np.array([
                [1 - poisson, poisson, 0],
                [poisson, 1 - poisson, 0],
                [0, 0, (1 - 2 * poisson) / 2]
            ])
            E = coeff * matrix
        elif condition == "plane stress":
            coeff = young / (1 - poisson ** 2)
            matrix = np.array([
                [1, poisson, 0],
                [poisson, 1, 0],
                [0, 0, (1-poisson)/2]
            ])
            E = coeff * matrix

        E_matrices[physical_tag] = {}
        E_matrices[physical_tag]["name"] = key
        E_matrices[physical_tag]["E"] = E

    return E_matrices


def gauss_quadrature(data):
    if data["element type"] == "T3":
        if data["integration points"] == 1:
            weights = np.array([1])
            locations = np.array([(1/3, 1/3)])
        elif data["integration points"] == 3:  # only bulk rule
            weights = np.array([1/3, 1/3, 1/3])
            locations = np.array(
                [
                    (1/6, 1/6),
                    (2/3, 1/6),
                    (1/6, 2/3),
                ]
            )
    elif data["element type"] == "T6":
        print("ERROR -- T6 element not implemented!!!")
        sys.exit()
    return weights, locations


x = lambda a, i, j: a[i][0] - a[j][0]
y = lambda b, i, j: b[i][1] - b[j][1]


def stiffness_matrix(e, data, mesh, E_matrices):

    t = data["thickness"]
    element_nodes = mesh.cells["triangle"][e]
    # print("nodes:\n", element_nodes)
    c = mesh.points[:,:2][element_nodes]
    # print("coord:\n", c)

    element_material = mesh.cell_data["triangle"]["gmsh:physical"][e]
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
        # print("j",j)
        B = (1/j) * np.array([
            (y(c, 1, 2), 0, y(c, 2, 0), 0, y(c, 0, 1), 0),
            (0, x(c, 2, 1), 0, x(c, 0, 2), 0 , x(c, 1, 0)),
            (x(c, 2, 1), y(c, 1, 2), x(c, 0, 2), y(c, 2, 0), x(c, 1, 0), y(c, 0, 1))
        ])
        # print("B", B)
        k_integral = B.T @ E @ B * t * 0.5 * j * w
        k += k_integral

    return k


def compute_global_dof(e, mesh):
    element_nodes = mesh.cells["triangle"][e]
    element_dof = np.zeros(6, dtype=np.int32)  # becomes 12 for T6
    for n in range(element_nodes.shape[0]):  # TODO check if applicable for BC
        element_dof[n*2] = element_nodes[n] * 2
        element_dof[n*2+1] = element_nodes[n] * 2 + 1
    return element_dof


def assembly(e, data, mesh, E_matrices, K):    
    k = stiffness_matrix(e, data, mesh, E_matrices)
    element_dof = compute_global_dof(e, mesh)

    for i in range(6):  # becomes 12 for T6
        I = element_dof[i]
        for j in range(6):  # becomes 12 for T6
            J = element_dof[j]
            K[I, J] += k[i, j]
    return K
