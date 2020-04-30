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


logger = logging.getLogger(__name__)


def base_analysis(mesh, element_type):
    # MESH
    # mesh = meshio.read(mesh_path)
    elements = mesh.cells_dict[element_type]
    nodal_coord = mesh.points[:,:2]
    num_elements = elements.shape[0]
    num_nodes = nodal_coord.shape[0]
    material_map = mesh.cell_data_dict["gmsh:physical"][element_type] - 1  # element-material map
    logger.debug("mesh info: %d elements, %d nodes", num_elements, num_nodes)
    # DATA
    integration_points = 1  # hard-coded but could be removed
    load_condition = "plane strain"  # TODO also this could be removed 'cause we need only plane strain case
    thickness = 1  # TODO make this a passed argument?
    # MATERIAL
    matrix = base.Material("matrix", 3.2, 0.35, load_condition)
    fiber = base.Material("fiber", 20, 0.20, load_condition)

    # BOUNDARY CONDITIONS INSTANCES
    left_side = bc.DirichletBC("left side", mesh, [0], 0.0)
    bl_corner = bc.DirichletBC("bottom left corner", mesh, [1], 0.0)
    right_side = bc.DirichletBC("right side", mesh, [0], 1.0)

    # ASSEMBLY
    E_material = base.compute_E_material(num_elements, material_map, mesh.field_data, matrix, fiber)
    K = np.zeros((num_nodes * 2, num_nodes * 2))
    R = np.zeros(num_nodes * 2)
    K = base.assembly(K, num_elements, elements, nodal_coord, material_map, E_material, thickness, element_type, integration_points)
    
    # contrained dof rows of K are saved now
    reaction_dof = bc.dirichlet_dof(left_side, bl_corner)
    K_rows = K[reaction_dof, :]

    # BOUNDARY CONDITIONS APPLICATION
    K, R = bc.apply_dirichlet(K, R, left_side, bl_corner, right_side)
    # SOLVER
    D = np.linalg.solve(K, R)
    reactions = np.dot(K_rows, D)
    modulus = base.compute_modulus(nodal_coord, right_side, reactions, thickness)
    logger.debug("E2 = %f", modulus)

    return modulus


def sp_base_analysis(mesh, element_type):
    # MESH
    # mesh = meshio.read(mesh_path)
    elements = mesh.cells_dict[element_type]
    nodal_coord = mesh.points[:,:2]
    num_elements = elements.shape[0]
    num_nodes = nodal_coord.shape[0]
    material_map = mesh.cell_data_dict["gmsh:physical"][element_type] - 1  # element-material map
    logger.debug("mesh info: %d elements, %d nodes", num_elements, num_nodes)
    # DATA
    integration_points = 1  # hard-coded but could be removed
    load_condition = "plane strain"  # TODO also this could be removed 'cause we need only plane strain case
    thickness = 1  # TODO make this a passed argument?
    # MATERIAL
    matrix = base.Material("matrix", 3.2, 0.35, load_condition)
    fiber = base.Material("fiber", 20, 0.20, load_condition)

    # BOUNDARY CONDITIONS INSTANCES
    left_side = bc.DirichletBC("left side", mesh, [0], 0.0)
    bl_corner = bc.DirichletBC("bottom left corner", mesh, [1], 0.0)
    right_side = bc.DirichletBC("right side", mesh, [0], 1.0)

    # ASSEMBLY
    E_material = base.compute_E_material(num_elements, material_map, mesh.field_data, matrix, fiber)
    K = sparse.csc_matrix((2 * num_nodes, 2 * num_nodes))
    R = np.zeros(num_nodes * 2)
    K = base.sp_assembly(K, num_elements, num_nodes, elements, nodal_coord, material_map, E_material, thickness, element_type, integration_points)
    
    # save constrained dof rows of K
    # dirichlet dof are built only for boundaries with related reactions
    dirichlet_dof, dirichlet_values = bc.build_dirichlet_data(left_side, bl_corner)
    K = K.tocsr()
    K_rows = K[dirichlet_dof,:]
    K = K.tocsc()

    # BOUNDARY CONDITIONS APPLICATION
    K, R = bc.sp_apply_dirichlet(num_nodes, K, R, left_side, bl_corner, right_side)
    # SOLVER
    D = linalg.spsolve(K, R)
    reactions = K_rows.dot(D)
    modulus = base.compute_modulus(nodal_coord, right_side, reactions, thickness)
    logger.debug("E2 = %f", modulus)

    return modulus


def vector_analysis(mesh, element_type, post_process=False, vtk_filename=None):
    # MESH
    # mesh = meshio.read(mesh_path)
    elements = mesh.cells_dict[element_type]
    nodal_coord = mesh.points[:,:2]
    num_elements = elements.shape[0]
    num_nodes = nodal_coord.shape[0]
    material_map = mesh.cell_data_dict["gmsh:physical"][element_type] - 1  # element-material map
    logger.debug("mesh info: %d elements, %d nodes", num_elements, num_nodes)
    # DATA
    load_condition = "plane strain"  # "plane stress" or "plane strain"
    thickness = 1
    # MATERIAL
    matrix = base.Material("matrix", 10, 0.3, load_condition)
    fiber = base.Material("fiber", 100, 0.3, load_condition)

    # BOUNDARY CONDITIONS INSTANCES
    left_side = bc.DirichletBC("left side", mesh, [0], 0.0)
    bl_corner = bc.DirichletBC("bottom left corner", mesh, [1], 0.0)
    right_side = bc.DirichletBC("right side", mesh, [0], 1.0)

    # ASSEMBLY
    E_array = vector.compute_E_array(num_elements, material_map, mesh.field_data, matrix, fiber)
    R = np.zeros(num_nodes * 2)
    K = vector.assembly(num_elements, num_nodes, elements, nodal_coord, E_array, thickness)

    # save constrained dof rows of K
    # dirichlet dof are built only for boundaries with related reactions
    dirichlet_dof, dirichlet_values = bc.build_dirichlet_data(left_side, bl_corner)
    K = K.tocsr()
    K_rows = K[dirichlet_dof,:]
    K = K.tocsc()
    
    # BOUNDARY CONDITIONS APPLICATION
    K, R = bc.sp_apply_dirichlet(num_nodes, K, R, left_side, bl_corner, right_side)
    # SOLVER
    D = linalg.spsolve(K, R)
    reactions = K_rows.dot(D)
    modulus = base.compute_modulus(nodal_coord, right_side, reactions, thickness)
    logger.debug("E2 = %f", modulus)

    if post_process:
        D_ready = np.column_stack((D[::2], D[1::2], np.zeros(num_nodes)))
        D_dict = {"displacement": D_ready}
        mesh.point_data = D_dict
        mesh.write(f"../data/vtk/{vtk_filename}.vtk")
        logger.debug("VTK file created")

    return modulus


if __name__ == "__main__":
    # logger
    log_lvl = logging.DEBUG
    root_logger = logging.getLogger()
    root_logger.setLevel(log_lvl)
    handler = logging.StreamHandler()
    handler.setLevel(log_lvl)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    np.set_printoptions(linewidth=200)
    
    start_time = time.time()
    mesh = meshio.read("../data/msh/terada_30F.msh")
    # E = base_analysis(mesh, "triangle")
    # E = sp_base_analysis(mesh, "triangle")
    E = vector_analysis(mesh, "triangle", post_process=True, vtk_filename="terada_30F")
    print(f"--- {time.time() - start_time} seconds ---")