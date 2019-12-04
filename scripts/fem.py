import json
import sys
import time

import meshio
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from feat.boundary import DirichletBC, NeumannBC, dirichlet_dof
from feat.helpers import (assembly, compute_E_matrices, gauss_quadrature,
                     stiffness_matrix)
from feat.post_proc import compute_modulus
from feat.vect_helpers import vect_assembly

def analysis(): 
    
    # SETTINGS
    data_path = "../data/test.json"
    mesh_path = "../gmsh/msh/test.msh"
    BASE = False
    VECT = True
    
    # DATA
    with open(data_path, "r") as data_file:
            data = json.load(data_file)
    weights, locations = gauss_quadrature(data)
    thickness = data["thickness"]

    # MESH
    mesh = meshio.read(mesh_path)
    elements_num = mesh.cells["triangle"].shape[0]
    nodes = mesh.points.shape[0]

    # BOUNDARY CONDITIONS INSTANCES
    left_side = DirichletBC("left side", data, mesh)
    br_corner = DirichletBC("bottom right corner", data, mesh)
    tr_corner = NeumannBC("top right corner", data, mesh)

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
        K, K_stored, R = vect_assembly(data, mesh, left_side, br_corner)
        print("R:\n", R)
        print()

    # SOLVER
    if BASE:
        left_side.impose(K, R)
        br_corner.impose(K, R)
        tr_corner.impose(R)
        print("K:\n", K)
        print("R:\n", R)
        print()

        D = np.linalg.solve(K, R)
        print("D:\n", D)
        print()

    elif VECT:
        tr_corner.impose(R)
        print("R:\n", R)
        print()

        D = linalg.spsolve(K, R)
        print("D:\n", D)
        print()

    
    return D


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    start_time = time.time()
    analysis()
    print(f"--- {time.time() - start_time} seconds ---")
