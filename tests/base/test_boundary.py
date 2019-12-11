import meshio
import numpy as np
import pytest

from feat import base
from feat.base import DirichletBC, NeumannBC


def test_build_dirichlet_data():
    
    mesh_path = "tests/data/msh/base.msh"
    mesh = meshio.read(mesh_path)

    left_side = DirichletBC("left side", mesh, [0], 0.0)
    bl_corner = DirichletBC("bottom left corner", mesh, [1], 0.0)
    right_side = DirichletBC("right side", mesh, [0], 1.0)

    dirichlet_dof, dirichlet_values = base.build_dirichlet_data(left_side, bl_corner, right_side)
    dir_dof_true = np.array([0, 6, 1, 2, 4])
    dir_values_true = np.array([0., 0., 0., 1., 1.])
    
    np.testing.assert_allclose(dir_dof_true, dirichlet_dof)
    np.testing.assert_allclose(dir_values_true, dirichlet_values)

