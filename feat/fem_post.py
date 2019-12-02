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
    data_path = "../data/base.json"
    mesh_path = "../gmsh/msh/base.msh"
    POST = True
    BASE = True
    VECT = False
    
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
    bl_corner = DirichletBC("bottom left corner", data, mesh)
    right_side = DirichletBC("right side", data, mesh)

    if BASE:
        E_matrices = compute_E_matrices(data, mesh)
        K = np.zeros((nodes * 2, nodes * 2))
        R = np.zeros(nodes * 2)
        for e in range(elements_num):  # number of elements_num
            K = assembly(e, data, mesh, E_matrices, K)
        print("K:\n", K)
        print("R:\n", R)
        print()
        if POST:
            # contrained dof rows of K are saved now
            reaction_dof = dirichlet_dof(left_side, bl_corner)
            K_rows = K[reaction_dof, :]

    elif VECT:
        K, K_stored, R = vect_assembly(data, mesh, left_side, bl_corner, right_side)
        print("K:\n", K.toarray())
        print("R:\n", R)
        print()
        if POST:
            reaction_dof = dirichlet_dof(left_side, bl_corner)
            K_rows = K[reaction_dof, :]
            print("K_rows\n", K_rows)

    # SOLVER
    if BASE:
        left_side.impose(K, R)
        bl_corner.impose(K, R)
        right_side.impose(K, R)
        print("K:\n", K)
        print("R:\n", R)
        print()

        D = np.linalg.solve(K, R)
        print("D:\n", D)
        print()

        if POST:
            reactions = np.dot(K_rows, D)
            print("reactions:\n", reactions)
            print()
            modulus = compute_modulus(mesh, right_side, reactions, thickness)
            print("modulus:\n", modulus)

    elif VECT:
        D = linalg.spsolve(K, R)
        print("D:\n", D)
        print()
        
        if POST:
            pass

    
    return D


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    start_time = time.time()
    analysis()
    print(f"--- {time.time() - start_time} seconds ---")
