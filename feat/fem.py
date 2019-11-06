import json
from pprint import pprint

import meshio
import numpy as np

from boundary import DirichletBC, NeumannBC, dirichlet_dof
from helpers import (assembly, compute_E_matrices, gauss_quadrature,
                     stiffness_matrix)
from post import compute_modulus


def analysis(): 
    # DATA
    with open(r'../data/base.json', "r") as data_file:
            data = json.load(data_file)

    element_type = data["element type"]
    ip_number = data["integration points"]
    thickness = data["thickness"]
    post = data["post-processing"]

    # NUMERICAL INTEGRATION
    weights, locations = gauss_quadrature(data)

    # MESH
    mesh = meshio.read(r"../gmsh/msh/base.msh")
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
    # pprint(E_matrices)
    # print()

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
            K = assembly(e, connectivity_table, k, K)

    # K_saved = np.copy(K)  # FIXME now this is useless

    print("K:\n", K)
    print("R:\n", R)
    print()


    left_side = DirichletBC("left side", data, mesh)
    left_corner = DirichletBC("bottom left corner", data, mesh)
    right_side = DirichletBC("right side", data, mesh)
    if post:
        # contrained dof rows of K are saved now
        reaction_dof = dirichlet_dof(left_side)
        K_rows = K[reaction_dof, :]

    left_side.impose(K, R)
    left_corner.impose(K, R)
    right_side.impose(K, R)
    print("K:\n", K)
    print("R:\n", R)
    print()

    # Solution of the system
    D = np.linalg.solve(K, R)
    print("D:\n", D)
    print()

    if post:
        reactions = np.dot(K_rows, D)
        print("reactions:\n", reactions)
        print()
        modulus = compute_modulus(mesh, right_side, reactions, thickness)
        print("modulus:\n", modulus)


if __name__ == "__main__":
    analysis()
