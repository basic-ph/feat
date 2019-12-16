import json
import sys
import time

import meshio
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from feat import base, vect
from feat.base import DirichletBC, NeumannBC


def main():
    # SETTINGS
    mesh_path = "./data/msh/base_large.msh"
    
    # DATA
    load_condition = "plane strain"  # "plane stress" or "plane strain"
    thickness = 1

    # MATERIAL
    cheese = base.Material(1, 70, 0.0, load_condition)

    # MESH
    mesh = meshio.read(mesh_path)
    elements_num = mesh.cells["triangle"].shape[0]
    nodes = mesh.points.shape[0]

    # BOUNDARY CONDITIONS INSTANCES
    left_side = DirichletBC("left side", mesh, [0], 0.0)
    bl_corner = DirichletBC("bottom left corner", mesh, [1], 0.0)
    right_side = DirichletBC("right side", mesh, [0], 1.0)

    # ASSEMBLY
    E_array = vect.compute_E_array(mesh, cheese)
    R = np.zeros(nodes * 2)
    K = vect.assembly(mesh, E_array, thickness)
    print("K:\n", K.toarray())
    print("R:\n", R)
    print()

    # save constrained dof rows of K
    # dirichlet dof are built only for boundaries with related reactions
    dirichlet_dof, dirichlet_values = base.build_dirichlet_data(left_side, bl_corner)
    K = K.tocsr()
    K_rows = K[dirichlet_dof,:]
    K = K.tocsc()
    
    # BOUNDARY CONDITIONS APPLICATION
    K, R = vect.apply_dirichlet(nodes, K, R, left_side, bl_corner, right_side)
    print("K:\n", K.toarray())
    print("R:\n", R)
    print()

    # SOLVER
    D = linalg.spsolve(K, R)
    print("D:\n", D)
    print()

    reactions = K_rows.dot(D)
    print("reactions:\n", reactions)
    print()
    modulus = base.compute_modulus(mesh, right_side, reactions, thickness)
    print("modulus:\n", modulus)



if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    start_time = time.time()
    main()
    print(f"--- {time.time() - start_time} seconds ---")
