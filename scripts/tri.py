import json
import sys
import time
import logging

import meshio
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from feat import base
from feat import boundary as bc


def main():

    logging.basicConfig(level=logging.DEBUG)

    # SETTINGS
    mesh_path = "./data/msh/tri.msh"

    # DATA
    element_type = "T3"
    integration_points = 1
    load_condition = "plane strain"  # "plane stress" or "plane strain"
    thickness = 1

    # MATERIAL
    rubber = base.Material(1, 10, 0.0, load_condition)

    # MESH
    mesh = meshio.read(mesh_path)
    elements_num = mesh.cells["triangle"].shape[0]
    nodes = mesh.points.shape[0]

    # BOUNDARY CONDITIONS INSTANCES
    node_1 = bc.DirichletBC("node 1", mesh, [0, 1], 0.0)
    node_3 = bc.DirichletBC("node 3", mesh, [0], 0.0)
    node_2_x = bc.NeumannBC("node 2", mesh, [0], 0.4)
    node_2_y = bc.NeumannBC("node 2", mesh, [1], 0.4)
    
    # ASSEMBLY
    E_array = base.compute_E_array(mesh, rubber)
    K = np.zeros((nodes * 2, nodes * 2))
    R = np.zeros(nodes * 2)
    K = base.assembly(K, elements_num, mesh, E_array, thickness, element_type, integration_points)
    print("K:\n", K)
    print("R:\n", R)
    print()

    # contrained dof rows of K are saved now
    reaction_dof = bc.dirichlet_dof(node_1, node_3)
    K_rows = K[reaction_dof, :]

    # BOUNDARY CONDITIONS APPLICATION
    K, R = bc.apply_dirichlet(K, R, node_1, node_3)
    R = bc.apply_neumann(R, node_2_x, node_2_y)
    print("K:\n", K)
    print("R:\n", R)
    print()

    # SOLVER
    D = np.linalg.solve(K, R)
    print("D:\n", D)
    print()

    reactions = np.dot(K_rows, D)
    print("reactions:\n", reactions)
    # print()
    # modulus = base.compute_modulus(mesh, right_side, reactions, thickness)
    # print("modulus:\n", modulus)


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    start_time = time.time()
    main()
    print(f"--- {time.time() - start_time} seconds ---")