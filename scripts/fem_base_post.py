import json
import sys
import time

import meshio
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from feat import base
from feat import boundary as bc


def main():
    # SETTINGS
    mesh_path = "./data/msh/base.msh"

    # DATA
    element_type = "T3"
    integration_points = 1
    load_condition = "plane strain"  # "plane stress" or "plane strain"
    thickness = 1

    # MATERIAL
    cheese = base.Material(1, 70, 0.0, load_condition)

    # MESH
    mesh = meshio.read(mesh_path)
    elements_num = mesh.cells["triangle"].shape[0]
    nodes = mesh.points.shape[0]

    # BOUNDARY CONDITIONS INSTANCES
    left_side = bc.DirichletBC("left side", mesh, [0], 0.0)
    bl_corner = bc.DirichletBC("bottom left corner", mesh, [1], 0.0)
    right_side = bc.DirichletBC("right side", mesh, [0], 1.0)

    # ASSEMBLY
    E_array = base.compute_E_array(mesh, cheese)
    K = np.zeros((nodes * 2, nodes * 2))
    R = np.zeros(nodes * 2)
    K = base.assembly(K, elements_num, mesh, E_array, thickness, element_type, integration_points)
    # print("K:\n", K)
    # print("R:\n", R)
    # print()

    # contrained dof rows of K are saved now
    reaction_dof = bc.dirichlet_dof(left_side, bl_corner)
    K_rows = K[reaction_dof, :]

    # BOUNDARY CONDITIONS APPLICATION
    K, R = bc.apply_dirichlet(K, R, left_side, bl_corner, right_side)
    # print("K:\n", K)
    # print("R:\n", R)
    # print()

    # SOLVER
    D = np.linalg.solve(K, R)
    print("D:\n", D)
    print()

    reactions = np.dot(K_rows, D)
    print("reactions:\n", reactions)
    print()
    modulus = base.compute_modulus(mesh, right_side, reactions, thickness)
    print("modulus:\n", modulus)


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    start_time = time.time()
    main()
    print(f"--- {time.time() - start_time} seconds ---")
