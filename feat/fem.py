import json
from pprint import pprint

import meshio
import numpy as np

from boundary import DirichletBC, NeumannBC, dirichlet_dof
from helpers import (assembly, compute_E_matrices, gauss_quadrature,
                     stiffness_matrix)
from post_proc import compute_modulus


def analysis(data_path, mesh_path): 
    # DATA
    with open(data_path, "r") as data_file:
            data = json.load(data_file)

    element_type = data["element type"]
    ip_number = data["integration points"]
    thickness = data["thickness"]
    post = data["post-processing"]

    # NUMERICAL INTEGRATION
    weights, locations = gauss_quadrature(data)

    # MESH
    mesh = meshio.read(mesh_path)
    nodal_coordinates = mesh.points[:,:2]  # slice is used to remove 3rd coordinate
    nodes = mesh.points.shape[0]
    dof = nodes * 2
    connectivity_table = mesh.cells["triangle"]
    elements = connectivity_table.shape[0]

    # this array contains material tag for every element in mesh
    element_material_map = mesh.cell_data["triangle"]["gmsh:physical"]
    # print(element_material_map)
    # print()

    E_matrices = compute_E_matrices(data, mesh)
    pprint(E_matrices)
    print()

    # arrays init
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
            print("k:\n",k)
            K = assembly(e, connectivity_table, k, K)

    print("K:\n", K)
    print("R:\n", R)
    print()


    left_side = DirichletBC("left side", data, mesh)
    br_corner = DirichletBC("bottom right corner", data, mesh)
    tr_corner = NeumannBC("top right corner", data, mesh)
    if post:
        # contrained dof rows of K are saved now
        reaction_dof = dirichlet_dof(left_side)
        K_rows = K[reaction_dof, :]

    left_side.impose(K, R)
    br_corner.impose(K, R)
    tr_corner.impose(R)
    print("K:\n", K)
    print("R:\n", R)
    print()

    # Solution of the system
    D = np.linalg.solve(K, R)
    print("D:\n", D)
    print()
    # print(D[2], D[4], D[5])
    # print()
    # print(K[2])
    # print(K[4])
    # print(K[5])

    return D

    if post:
        reactions = np.dot(K_rows, D)
        print("reactions:\n", reactions)
        print()
        modulus = compute_modulus(mesh, right_side, reactions, thickness)
        print("modulus:\n", modulus)


if __name__ == "__main__":
    # np.set_printoptions(precision=2)
    data_path = "../data/test.json"
    mesh_path = "../gmsh/msh/test.msh"
    analysis(data_path, mesh_path)
