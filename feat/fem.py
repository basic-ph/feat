import json
from pprint import pprint

import meshio
import numpy as np

from boundary import DirichletBC, NeumannBC, dirichlet_dof
from helpers import (assembly, compute_E_matrices, gauss_quadrature,
                     stiffness_matrix)
from post_proc import compute_modulus


def analysis(): 
    
    # SETTINGS
    data_path = "../data/test.json"
    mesh_path = "../gmsh/msh/test.msh"
    POST = False
    
    # DATA
    with open(data_path, "r") as data_file:
            data = json.load(data_file)
    weights, locations = gauss_quadrature(data)

    # MESH
    mesh = meshio.read(mesh_path)
    E_matrices = compute_E_matrices(data, mesh)

    K = np.zeros((mesh.points.shape[0] * 2, mesh.points.shape[0] * 2))
    R = np.zeros(mesh.points.shape[0] * 2)

    for e in range(mesh.cells["triangle"].shape[0]):  # number of elements
        K = assembly(e, data, mesh, E_matrices, K)

    print("K:\n", K)
    print("R:\n", R)
    print()

    # BOUNDARY CONDITIONS INSTANCES
    left_side = DirichletBC("left side", data, mesh)
    br_corner = DirichletBC("bottom right corner", data, mesh)
    tr_corner = NeumannBC("top right corner", data, mesh)
    if POST:
        # contrained dof rows of K are saved now
        reaction_dof = dirichlet_dof(left_side)
        K_rows = K[reaction_dof, :]

    left_side.impose(K, R)
    br_corner.impose(K, R)
    tr_corner.impose(R)
    print("K:\n", K)
    print("R:\n", R)
    print()

    # SOLVER
    D = np.linalg.solve(K, R)
    print("D:\n", D)
    print()

    if POST:
        reactions = np.dot(K_rows, D)
        print("reactions:\n", reactions)
        print()
        modulus = compute_modulus(mesh, right_side, reactions, thickness)
        print("modulus:\n", modulus)
    
    return D


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    analysis()
