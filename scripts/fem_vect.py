import json
import sys
import time

import meshio
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from feat import base
from feat import boundary as bc
from feat import vector


def main():
    # SETTINGS
    mesh_path = "./data/msh/test.msh"
    
    # DATA
    load_condition = "plane stress"  # "plane stress" or "plane strain"
    thickness = 0.5

    # MATERIAL
    steel = base.Material(1, 3e7, 0.25, load_condition)

    # MESH
    mesh = meshio.read(mesh_path)
    elements_num = mesh.cells["triangle"].shape[0]
    nodes = mesh.points.shape[0]

    # BOUNDARY CONDITIONS INSTANCES
    left_side = bc.DirichletBC("left side", mesh, [0, 1], 0.0)
    br_corner = bc.DirichletBC("bottom right corner", mesh, [1], 0.0)
    tr_corner = bc.NeumannBC("top right corner", mesh, [1], -1000.0)

    # ASSEMBLY
    E_array = vector.compute_E_array(mesh, steel)
    R = np.zeros(nodes * 2)
    K = vector.assembly(mesh, E_array, thickness)
    print("K:\n", K.toarray())
    print("R:\n", R)
    print()

    # BOUNDARY CONDITIONS APPLICATION
    K, R = bc.sp_apply_dirichlet(nodes, K, R, left_side, br_corner)
    R = bc.apply_neumann(R, tr_corner)
    print("K:\n", K.toarray())
    print("R:\n", R)
    print()

    # SOLVER
    D = linalg.spsolve(K, R)
    print("D:\n", D)
    print()


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    start_time = time.time()
    main()
    print(f"--- {time.time() - start_time} seconds ---")
