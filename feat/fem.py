import json
import sys
import time

import meshio
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from boundary import DirichletBC, NeumannBC, dirichlet_dof
from helpers import (assembly, compute_E_matrices, gauss_quadrature,
                     stiffness_matrix)
from post_proc import compute_modulus
from vect_helpers import vect_assembly

def analysis(): 
    
    # SETTINGS
    data_path = "../data/test.json"
    mesh_path = "../gmsh/msh/test.msh"
    POST = False
    BASE = False
    VECT = True
    
    # DATA
    with open(data_path, "r") as data_file:
            data = json.load(data_file)
    weights, locations = gauss_quadrature(data)

    # MESH
    mesh = meshio.read(mesh_path)
    elements_num = mesh.cells["triangle"].shape[0]
    nodes = mesh.points.shape[0]

    if BASE:
        E_matrices = compute_E_matrices(data, mesh)
        K = np.zeros((nodes * 2, nodes * 2))
        R = np.zeros(nodes * 2)
        for e in range(elements_num):  # number of elements_num
            K = assembly(e, data, mesh, E_matrices, K)
        print("K:\n", K)
        print("R:\n", R)
        print()
    elif VECT:
        R = np.zeros(nodes * 2)
        K_array, I_array, J_array = vect_assembly(data, mesh)
        K = sparse.csr_matrix(
            (
                np.ravel(K_array),  # data
                (np.ravel(I_array), np.ravel(J_array)),  # row_ind, col_ind
            ),
            shape=(2 * nodes, 2 * nodes),
        )

        K = K.tolil()  # convert to lil matrix TODO
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


    # SOLVER
    if BASE:
        left_side.impose(K, R)
        br_corner.impose(K, R)
        tr_corner.impose(R)
        # print("K:\n", K)
        print("R:\n", R)
        print()

        D = np.linalg.solve(K, R)
        print("D:\n", D)
        print()
    elif VECT:
        left_side.sparse_impose(K, R)
        br_corner.sparse_impose(K, R)
        tr_corner.impose(R)
        # print("K:\n", K)
        print("R:\n", R)
        print()

        K = K.tocsr()
        D = linalg.spsolve(K, R)
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
    start_time = time.time()
    analysis()
    print(f"--- {time.time() - start_time} seconds ---")
