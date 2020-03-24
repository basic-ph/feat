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
    # LOGGING
    # # main_log = logging.getLogger(__name__)
    # # main_log.setLevel(logging.INFO)
    # main_handler = logging.StreamHandler()  # # main_log handler
    # main_handler.setLevel(logging.INFO)
    # main_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # # main_log formatter
    # main_handler.setFormatter(main_formatter)
    # # main_log.addHandler(main_handler)

    feat_log_lvl = logging.DEBUG
    feat_log = logging.getLogger("feat")
    feat_log.setLevel(feat_log_lvl)
    feat_handler = logging.StreamHandler()
    feat_handler.setLevel(feat_log_lvl)
    feat_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    feat_handler.setFormatter(feat_formatter)
    feat_log.addHandler(feat_handler)
    
    # SETTINGS
    mesh_path = "./data/msh/base.msh"
    # main_log.info("MESH FILE: %s", mesh_path)

    # DATA
    element_type = "triangle"
    integration_points = 1
    load_condition = "plane strain"  # "plane stress" or "plane strain"
    thickness = 1
    # main_log.info("LOAD CONDITION: %s", load_condition)
    # main_log.info("THICKNESS: %s", thickness)

    # MATERIAL
    cheese = base.Material("cheese", 70, 0.3, load_condition)
    # main_log.info("MATERIALS: TODO")

    # MESH
    mesh = meshio.read(mesh_path)
    elements_num = mesh.cells_dict[element_type].shape[0]
    nodes = mesh.points.shape[0]
    # main_log.info("MESH INFO: %d elements, %d nodes", elements_num, nodes)

    # BOUNDARY CONDITIONS INSTANCES
    left_side = bc.DirichletBC("left side", mesh, [0], 0.0)
    bl_corner = bc.DirichletBC("bottom left corner", mesh, [1], 0.0)
    right_side = bc.DirichletBC("right side", mesh, [0], 1.0)
    # main_log.info("BOUNDARY CONDITIONS: TODO")

    # ASSEMBLY
    E_array = base.compute_E_array(mesh, element_type, cheese)
    # main_log.info("E_array:\n %s", E_array)
    K = np.zeros((nodes * 2, nodes * 2))
    R = np.zeros(nodes * 2)
    K = base.assembly(K, elements_num, mesh, E_array, thickness, element_type, integration_points)
    # main_log.debug("STIFFNESS MATRIX (K) BEFORE BC:\n %s\n", K)

    # contrained dof rows of K are saved now
    reaction_dof = bc.dirichlet_dof(left_side, bl_corner)
    K_rows = K[reaction_dof, :]

    # BOUNDARY CONDITIONS APPLICATION
    K, R = bc.apply_dirichlet(K, R, left_side, bl_corner, right_side)
    # main_log.debug("STIFFNESS MATRIX (K) AFTER BC:\n %s\n", K)
    # main_log.debug("LOAD VECTOR (R) BEFORE BC:\n %s\n", R)

    # SOLVER
    D = np.linalg.solve(K, R)
    # main_log.info("DISPLACEMENTS VECTOR (D):\n %s\n", D)

    reactions = np.dot(K_rows, D)
    # main_log.debug("REACTIONS (dirichlet dofs):\n %s\n", reactions)
    modulus = base.compute_modulus(mesh, right_side, reactions, thickness)
    # main_log.info("RESULTING ELASTIC MODULUS: %f", modulus)


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    start_time = time.time()
    main()
    print(f"--- {time.time() - start_time} seconds ---")