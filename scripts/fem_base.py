import json
import sys
import time

import meshio
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from feat import base
from feat.base import DirichletBC, NeumannBC


def main():
    # SETTINGS
    mesh_path = "./data/msh/test.msh"

    # DATA
    element_type = "T3"
    integration_points = 1
    load_condition = "plane stress"  # "plane stress" or "plane strain"
    thickness = 0.5

    # MATERIAL
    steel = base.Material(1, 3e7, 0.25, load_condition)

    # MESH
    mesh = meshio.read(mesh_path)
    elements_num = mesh.cells["triangle"].shape[0]
    nodes = mesh.points.shape[0]

    # BOUNDARY CONDITIONS INSTANCES
    left_side = DirichletBC("left side", mesh, [0, 1], 0.0)
    br_corner = DirichletBC("bottom right corner", mesh, [1], 0.0)
    tr_corner = NeumannBC("top right corner", mesh, [1], -1000.0)

    # ASSEMBLY
    E_array = base.compute_E_array(mesh, steel)
    K = np.zeros((nodes * 2, nodes * 2))
    R = np.zeros(nodes * 2)
    for e in range(elements_num):
        base.assembly(K, e, mesh, E_array, thickness, element_type, integration_points)
    print("K:\n", K)
    print("R:\n", R)
    print()

    # BOUNDARY CONDITIONS APPLICATION
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


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    start_time = time.time()
    main()
    print(f"--- {time.time() - start_time} seconds ---")
