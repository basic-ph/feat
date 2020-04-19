"""This module contains tests regarding simple problems resolved using FEAP
software from University of California, Berkeley. Results obtained with
this program are used as validation comparing them with those obtained
using feat python code.
This file is used for testing the base module.
"""

import logging

import meshio
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg

from feat import base
from feat import boundary as bc
from feat import vector


def test_feap_1():
    #LOGGING
    main_log = logging.getLogger(__name__)
    main_log.setLevel(logging.DEBUG)
    main_handler = logging.StreamHandler()  # main_log handler
    main_handler.setLevel(logging.DEBUG)
    main_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # main_log formatter
    main_handler.setFormatter(main_formatter)
    main_log.addHandler(main_handler)

    feat_log = logging.getLogger("feat")
    feat_log.setLevel(logging.DEBUG)
    feat_handler = logging.StreamHandler()
    feat_handler.setLevel(logging.DEBUG)
    feat_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    feat_handler.setFormatter(feat_formatter)
    feat_log.addHandler(feat_handler)
    
    # SETTINGS
    mesh_path = "tests/data/msh/feap_1.msh"
    main_log.info("MESH FILE: %s", mesh_path)

    # DATA
    element_type = "triangle"
    integration_points = 1
    load_condition = "plane strain"  # "plane stress" or "plane strain"
    thickness = 1
    main_log.info("LOAD CONDITION: %s", load_condition)
    main_log.info("THICKNESS: %s", thickness)

    # MATERIAL
    cheese = base.Material("cheese", 70, 0.3, load_condition)  #FIXME
    main_log.info("MATERIALS: TODO")

    # MESH
    mesh = meshio.read(mesh_path)
    elements = mesh.cells_dict[element_type]
    nodal_coord = mesh.points[:,:2]
    print(type(nodal_coord))
    print(nodal_coord)
    num_elements = elements.shape[0]
    num_nodes = nodal_coord.shape[0]
    material_map = mesh.cell_data_dict["gmsh:physical"][element_type] - 1  # element-material map
    main_log.info("MESH INFO: %d elements, %d nodes", num_elements, num_nodes)

    # BOUNDARY CONDITIONS INSTANCES
    left_side = bc.DirichletBC("left side", mesh, [0], 0.0)
    bl_corner = bc.DirichletBC("bottom left corner", mesh, [1], 0.0)
    right_side = bc.DirichletBC("right side", mesh, [0], 1.0)
    main_log.info("BOUNDARY CONDITIONS: TODO")

    # ASSEMBLY
    E_material = base.compute_E_material(num_elements, material_map, mesh.field_data, cheese)
    K = np.zeros((num_nodes * 2, num_nodes * 2))
    R = np.zeros(num_nodes * 2)
    K = base.assembly(K, num_elements, elements, nodal_coord, material_map, E_material, thickness, element_type, integration_points)
    main_log.debug("STIFFNESS MATRIX (K) BEFORE BC:\n %s\n", K)

    # contrained dof rows of K are saved now
    reaction_dof = bc.dirichlet_dof(left_side, bl_corner)
    K_rows = K[reaction_dof, :]

    # BOUNDARY CONDITIONS APPLICATION
    K, R = bc.apply_dirichlet(K, R, left_side, bl_corner, right_side)
    main_log.debug("STIFFNESS MATRIX (K) AFTER BC:\n %s\n", K)
    main_log.debug("LOAD VECTOR (R) BEFORE BC:\n %s\n", R)

    # SOLVER
    D = np.linalg.solve(K, R)
    main_log.info("DISPLACEMENTS VECTOR (D):\n %s\n", D)

    reactions = np.dot(K_rows, D)
    main_log.debug("REACTIONS (dirichlet dofs):\n %s\n", reactions)
    modulus = base.compute_modulus(nodal_coord, right_side, reactions, thickness)
    main_log.info("RESULTING ELASTIC MODULUS: %f", modulus)

    comparable_dofs = [0, 1, 2, 4, 5, 6, 7]
    D_true = np.array([
        0.0, 0.0,
        1.0, np.NaN,
        1.0, -4.28571429e-01,
        0.0, -4.28571429e-01,
    ])
    reactions_true = np.array([-3.84615385e+01, -3.84615385e+01, -7.10542736e-15])
    
    np.testing.assert_allclose(reactions_true, reactions)
    np.testing.assert_allclose(D_true[comparable_dofs], D[comparable_dofs])


@pytest.mark.parametrize(
    "poisson,D_true,reactions_true",
    [
        (
            0.3,
            np.array([0.0, 0.0, 7.28e-02, 2.76e-01, 0.0, 8.0e-03]),
            np.array([-0.64, 0.24, -0.4]),
        ),
        (
            0.0,
            np.array([0.0, 0.0, 1.04e-01, 2.3815385e-01, 0.0, 4.307692e-02]),
            np.array([-0.64, 0.24, -0.4]),
        )
    ],
)
def test_feap_2(poisson, D_true, reactions_true):
    # LOGGING
    main_log = logging.getLogger(__name__)
    main_log.setLevel(logging.DEBUG)
    main_handler = logging.StreamHandler()  # main_log handler
    main_handler.setLevel(logging.DEBUG)
    main_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # main_log formatter
    main_handler.setFormatter(main_formatter)
    main_log.addHandler(main_handler)

    feat_log = logging.getLogger("feat")
    feat_log.setLevel(logging.DEBUG)
    feat_handler = logging.StreamHandler()
    feat_handler.setLevel(logging.DEBUG)
    feat_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    feat_handler.setFormatter(feat_formatter)
    feat_log.addHandler(feat_handler)
    
    # SETTINGS
    mesh_path = "tests/data/msh/feap_2.msh"
    main_log.info("MESH FILE: %s", mesh_path)

    # DATA
    element_type = "triangle"
    integration_points = 1
    load_condition = "plane strain"  # "plane stress" or "plane strain"
    thickness = 1
    main_log.info("LOAD CONDITION: %s", load_condition)
    main_log.info("THICKNESS: %s", thickness)

    # MATERIAL
    rubber = base.Material("rubber", 10, poisson, load_condition)  #FIXME
    main_log.info("MATERIALS: TODO")

    # MESH
    mesh = meshio.read(mesh_path)
    elements = mesh.cells_dict[element_type]
    nodal_coord = mesh.points[:,:2]
    num_elements = elements.shape[0]
    num_nodes = nodal_coord.shape[0]
    material_map = mesh.cell_data_dict["gmsh:physical"][element_type] - 1  # element-material map
    main_log.info("MESH INFO: %d elements, %d nodes", num_elements, num_nodes)

    # BOUNDARY CONDITIONS INSTANCES
    left_side = bc.DirichletBC("left side", mesh, [0], 0.0)
    b_corner = bc.DirichletBC("bottom corner", mesh, [1], 0.0)
    r_corner_x = bc.NeumannBC("right corner", mesh, [0], 0.4)
    r_corner_y = bc.NeumannBC("right corner", mesh, [1], 0.4)
    main_log.info("BOUNDARY CONDITIONS: TODO")

    # ASSEMBLY
    E_material = base.compute_E_material(num_elements, material_map, mesh.field_data, rubber)
    main_log.debug("E array:\n %s\n", E_material)
    K = np.zeros((num_nodes * 2, num_nodes * 2))
    R = np.zeros(num_nodes * 2)
    K = base.assembly(K, num_elements, elements, nodal_coord, material_map, E_material, thickness, element_type, integration_points)
    main_log.debug("STIFFNESS MATRIX (K) BEFORE BC:\n %s\n", K)

    # contrained dof rows of K are saved now
    reaction_dof = bc.dirichlet_dof(left_side, b_corner)
    K_rows = K[reaction_dof, :]

    # BOUNDARY CONDITIONS APPLICATION
    K, R = bc.apply_dirichlet(K, R, left_side, b_corner)
    R = bc.apply_neumann(R, r_corner_x, r_corner_y)
    main_log.debug("STIFFNESS MATRIX (K) AFTER BC:\n %s\n", K)
    main_log.debug("LOAD VECTOR (R) BEFORE BC:\n %s\n", R)

    # SOLVER
    D = np.linalg.solve(K, R)
    main_log.info("DISPLACEMENTS VECTOR (D):\n %s\n", D)

    reactions = np.dot(K_rows, D)
    main_log.debug("REACTIONS (dirichlet dofs):\n %s\n", reactions)

    np.testing.assert_allclose(D_true, D)
    np.testing.assert_allclose(reactions_true, reactions)