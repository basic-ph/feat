"""This module contains tests regarding simple problems resolved using FEAP
software from University of California, Berkeley. Results obtained with
this program are used as validation comparing them with those obtained
using feat python code.
This file is used for testing the vector module.
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
    # LOGGING
    main_log = logging.getLogger(__name__)
    main_log.setLevel(logging.INFO)
    main_handler = logging.StreamHandler()  # main_log handler
    main_handler.setLevel(logging.INFO)
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
    load_condition = "plane strain"  # "plane stress" or "plane strain"
    thickness = 1
    main_log.info("LOAD CONDITION: %s", load_condition)
    main_log.info("THICKNESS: %s", thickness)

    # MATERIAL
    cheese = base.Material("cheese", 70, 0.3, load_condition)
    main_log.info("MATERIALS: TODO")

    # MESH
    mesh = meshio.read(mesh_path)
    elements_num = mesh.cells_dict["triangle"].shape[0]
    nodes = mesh.points.shape[0]
    main_log.info("MESH INFO: %d elements, %d nodes", elements_num, nodes)

    # BOUNDARY CONDITIONS INSTANCES
    left_side = bc.DirichletBC("left side", mesh, [0], 0.0)
    bl_corner = bc.DirichletBC("bottom left corner", mesh, [1], 0.0)
    right_side = bc.DirichletBC("right side", mesh, [0], 1.0)
    main_log.info("BOUNDARY CONDITIONS: TODO")

    # ASSEMBLY
    E_array = vector.compute_E_array(mesh, cheese)
    R = np.zeros(nodes * 2)
    K = vector.assembly(mesh, E_array, thickness)
    main_log.debug("STIFFNESS MATRIX (K) BEFORE BC:\n %s\n", K)

    # save constrained dof rows of K
    # dirichlet dof are built only for boundaries with related reactions
    dirichlet_dof, dirichlet_values = bc.build_dirichlet_data(left_side, bl_corner)
    K = K.tocsr()
    K_rows = K[dirichlet_dof,:]
    K = K.tocsc()
    
    # BOUNDARY CONDITIONS APPLICATION
    K, R = bc.sp_apply_dirichlet(nodes, K, R, left_side, bl_corner, right_side)
    main_log.debug("STIFFNESS MATRIX (K) AFTER BC:\n %s\n", K)
    main_log.debug("LOAD VECTOR (R) BEFORE BC:\n %s\n", R)

    # SOLVER
    D = linalg.spsolve(K, R)
    main_log.info("DISPLACEMENTS VECTOR (D):\n %s\n", D)

    reactions = K_rows.dot(D)
    main_log.debug("REACTIONS (dirichlet dofs):\n %s\n", reactions)
    modulus = base.compute_modulus(mesh, right_side, reactions, thickness)
    main_log.info("RESULTING ELASTIC MODULUS: %f", modulus)

    comparable_dofs = [2, 4, 5, 7]
    D_true = np.array([
        0.0, 0.0,
        1.0, np.NaN,
        1.0, -4.28571429e-01,
        0.0, -4.28571429e-01,
    ])
    np.testing.assert_allclose(D_true[comparable_dofs], D[comparable_dofs])


@pytest.mark.parametrize(
    "poisson,D_true,reactions_true",
    [
        (
            0.3,
            np.array([-3.06246839e-17, 4.54281749e-18, 7.28e-02, 2.76e-01, 0.0, 8.0e-03]),
            np.array([-0.64, 0.24, -0.4]),
        ),
        (
            0.0,
            np.array([2.01456792e-17, -2.37442191e-18, 1.04e-01, 2.38153846e-01, 0.0, 4.30769231e-02]),
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
    load_condition = "plane strain"  # "plane stress" or "plane strain"
    thickness = 1
    main_log.info("LOAD CONDITION: %s", load_condition)
    main_log.info("THICKNESS: %s", thickness)

    # MATERIAL
    rubber = base.Material("rubber", 10, poisson, load_condition)
    main_log.info("MATERIALS: TODO")

    # MESH
    mesh = meshio.read(mesh_path)
    elements_num = mesh.cells_dict["triangle"].shape[0]
    nodes = mesh.points.shape[0]
    main_log.info("MESH INFO: %d elements, %d nodes", elements_num, nodes)

    # BOUNDARY CONDITIONS INSTANCES
    left_side = bc.DirichletBC("left side", mesh, [0], 0.0)
    b_corner = bc.DirichletBC("bottom corner", mesh, [1], 0.0)
    r_corner_x = bc.NeumannBC("right corner", mesh, [0], 0.4)
    r_corner_y = bc.NeumannBC("right corner", mesh, [1], 0.4)
    main_log.info("BOUNDARY CONDITIONS: TODO")

    # ASSEMBLY
    E_array = vector.compute_E_array(mesh, rubber)
    main_log.debug("E array:\n %s\n", E_array)
    R = np.zeros(nodes * 2)
    K = vector.assembly(mesh, E_array, thickness)
    main_log.debug("STIFFNESS MATRIX (K) BEFORE BC:\n %s\n", K.todense())

    # save constrained dof rows of K
    # dirichlet dof are built only for boundaries with related reactions
    dirichlet_dof, dirichlet_values = bc.build_dirichlet_data(left_side, b_corner)
    K = K.tocsr()
    K_rows = K[dirichlet_dof,:]
    K = K.tocsc()
    
    # BOUNDARY CONDITIONS APPLICATION
    K, R = bc.sp_apply_dirichlet(nodes, K, R, left_side, b_corner)
    R = bc.apply_neumann(R, r_corner_x, r_corner_y)
    main_log.debug("STIFFNESS MATRIX (K) AFTER BC:\n %s\n", K.todense())
    main_log.debug("LOAD VECTOR (R) BEFORE BC:\n %s\n", R)

    # SOLVER
    D = linalg.spsolve(K, R)
    main_log.info("DISPLACEMENTS VECTOR (D):\n %s\n", D)

    reactions = K_rows.dot(D)
    main_log.debug("REACTIONS (dirichlet dofs):\n %s\n", reactions)

    np.testing.assert_allclose(D_true, D)
    np.testing.assert_allclose(reactions_true, reactions)