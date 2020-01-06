import meshio
import numpy as np
import pytest

from feat import boundary as bc


def test_build_dirichlet_data():
    
    mesh_path = "tests/data/msh/base.msh"
    mesh = meshio.read(mesh_path)

    left_side = bc.DirichletBC("left side", mesh, [0], 0.0)
    bl_corner = bc.DirichletBC("bottom left corner", mesh, [1], 0.0)
    right_side = bc.DirichletBC("right side", mesh, [0], 1.0)

    dirichlet_dof, dirichlet_values = bc.build_dirichlet_data(left_side, bl_corner, right_side)
    dir_dof_true = np.array([0, 6, 1, 2, 4])
    dir_values_true = np.array([0., 0., 0., 1., 1.])
    
    np.testing.assert_allclose(dir_dof_true, dirichlet_dof)
    np.testing.assert_allclose(dir_values_true, dirichlet_values)

