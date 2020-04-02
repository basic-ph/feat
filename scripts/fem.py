import json
import logging
import sys
import time

import meshio
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from feat import base, vector
from feat import boundary as bc


# LOGGING
logger = logging.getLogger(f"feat.{__name__}")


def base_analysis(mesh, element_type):
    # MESH
    # mesh = meshio.read(mesh_path)
    elements_num = mesh.cells_dict[element_type].shape[0]
    nodes = mesh.points.shape[0]
    logger.debug("mesh info: %d elements, %d nodes", elements_num, nodes)
    # DATA
    integration_points = 1  # hard-coded but could be removed
    load_condition = "plane strain"  # TODO also this could be removed 'cause we need only plane strain case
    thickness = 1  # TODO make this a passed argument?
    # MATERIAL
    matrix = base.Material("matrix", 100, 0.3, load_condition)
    fiber = base.Material("fiber", 700, 0.25, load_condition)

    # BOUNDARY CONDITIONS INSTANCES
    left_side = bc.DirichletBC("left side", mesh, [0], 0.0)
    bl_corner = bc.DirichletBC("bottom left corner", mesh, [1], 0.0)
    right_side = bc.DirichletBC("right side", mesh, [0], 1.0)

    # ASSEMBLY
    E_material = base.compute_E_material(mesh, element_type, matrix, fiber)
    K = np.zeros((nodes * 2, nodes * 2))
    R = np.zeros(nodes * 2)
    K = base.assembly(K, elements_num, mesh, E_material, thickness, element_type, integration_points)
    
    # contrained dof rows of K are saved now
    reaction_dof = bc.dirichlet_dof(left_side, bl_corner)
    K_rows = K[reaction_dof, :]

    # BOUNDARY CONDITIONS APPLICATION
    K, R = bc.apply_dirichlet(K, R, left_side, bl_corner, right_side)
    # SOLVER
    D = np.linalg.solve(K, R)
    reactions = np.dot(K_rows, D)
    modulus = base.compute_modulus(mesh, right_side, reactions, thickness)
    logger.debug("E2 = %f", modulus)

    return modulus


def vector_analysis(mesh, element_type):
    # MESH
    # mesh = meshio.read(mesh_path)
    elements_num = mesh.cells_dict[element_type].shape[0]
    nodes = mesh.points.shape[0]
    logger.debug("mesh info: %d elements, %d nodes", elements_num, nodes)
    # DATA
    load_condition = "plane strain"  # "plane stress" or "plane strain"
    thickness = 1
    # MATERIAL
    matrix = base.Material("matrix", 100, 0.3, load_condition)
    fiber = base.Material("fiber", 700, 0.25, load_condition)

    # BOUNDARY CONDITIONS INSTANCES
    left_side = bc.DirichletBC("left side", mesh, [0], 0.0)
    bl_corner = bc.DirichletBC("bottom left corner", mesh, [1], 0.0)
    right_side = bc.DirichletBC("right side", mesh, [0], 1.0)

    # ASSEMBLY
    E_array = vector.compute_E_array(mesh, element_type, matrix, fiber)
    R = np.zeros(nodes * 2)
    K = vector.assembly(mesh, element_type, E_array, thickness)

    # save constrained dof rows of K
    # dirichlet dof are built only for boundaries with related reactions
    dirichlet_dof, dirichlet_values = bc.build_dirichlet_data(left_side, bl_corner)
    K = K.tocsr()
    K_rows = K[dirichlet_dof,:]
    K = K.tocsc()
    
    # BOUNDARY CONDITIONS APPLICATION
    K, R = bc.sp_apply_dirichlet(nodes, K, R, left_side, bl_corner, right_side)
    # SOLVER
    D = linalg.spsolve(K, R)
    reactions = K_rows.dot(D)
    modulus = base.compute_modulus(mesh, right_side, reactions, thickness)
    logger.debug("E2 = %f", modulus)

    return modulus


if __name__ == "__main__":
    # logger
    feat_log_lvl = logging.DEBUG
    feat_log = logging.getLogger("feat")
    feat_log.setLevel(feat_log_lvl)
    feat_handler = logging.StreamHandler()
    feat_handler.setLevel(feat_log_lvl)
    feat_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    feat_handler.setFormatter(feat_formatter)
    feat_log.addHandler(feat_handler)
    
    np.set_printoptions(linewidth=200)
    
    start_time = time.time()
    mesh = meshio.read("./data/msh/rve_1.msh")
    # base_analysis(mesh, "triangle")
    vector_analysis(mesh, "triangle")
    # vector_analysis("./data/msh/rve_1.msh", "triangle")
    print(f"--- {time.time() - start_time} seconds ---")