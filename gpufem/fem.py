
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


def apply_bc(data, mesh, K, R):
    element_bc_map = mesh.cell_data["line"]["gmsh:physical"]
    print(element_bc_map)
    print()
    for key, value in data["bc"].items():
        # print(key)
        # print(value["name"])
        # print(value["type"])
        # print(value["dof"])
        # print("value", value["value"])
        # array containing indices of elements in a particular boundary
        boundary_elements = np.nonzero(element_bc_map == int(key))[0]
        boundary_connectivity = mesh.cells["line"][boundary_elements]
        # print(boundary_connectivity)
        boundary_nodes = np.lib.arraysetops.unique(boundary_connectivity)
        print("bound nodes:", boundary_nodes)
        for n in boundary_nodes:
            # print("node:", n)
            for d in value["dof"]:
                dof = n * 2 + d
                # print("dof:",dof)

                if value["type"] == 0:  # dirichlet
                    print("Dirichlet imposition...")
                    R -= value["value"] * K[:, dof]  # modify RHS
                    K[:, dof] = 0.0  # zero-out column
                    K[dof, :] = 0.0  # zero-out row
                    K[dof, dof] = 1.0  # set diagonal to 1
                    R[dof] = value["value"]  # enforce value
                elif value["type"] == 1:
                    print("Neumann imposition...")
                    node_count = boundary_nodes.shape[0]
                    nodal_load = value["value"] / node_count
                    R[dof] += nodal_load


def save_reaction_data(data, mesh, K):  # FIXME needs update after new bc classes
    reaction_dofs = []
    bc_data = data["bc"]
    element_bc_map = mesh.cell_data["line"]["gmsh:physical"]
    for key, value in data["bc"].items():
        if value["type"] == 0:  # dirichlet
            # array containing indices of elements in a particular boundary
            boundary_elements = np.nonzero(element_bc_map == int(key))[0]
            boundary_connectivity = mesh.cells["line"][boundary_elements]
            boundary_nodes = np.lib.arraysetops.unique(boundary_connectivity)
            for n in boundary_nodes:
                for d in value["dof"]:
                    dof = n * 2 + d
                    reaction_dofs.append(dof)
    saved_rows = K[reaction_dofs]
    return reaction_dofs, saved_rows