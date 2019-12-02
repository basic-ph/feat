import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import pytest

from feat.boundary import DirichletBC, NeumannBC, dirichlet_dof
from feat.helpers import gauss_quadrature
from feat.vect_helpers import (vect_assembly, vect_compute_E,
                               vect_compute_global_dof, vect_compute_K_entry)

# @pytest.mark.skip
def test_vect_fem(setup_data, setup_mesh):
    data = setup_data("data/test.json")
    weights, locations = gauss_quadrature(data)
    mesh = setup_mesh("gmsh/msh/test.msh")
    elements_num = mesh.cells["triangle"].shape[0]
    nodes = mesh.points.shape[0]

    left_side = DirichletBC("left side", data, mesh)
    br_corner = DirichletBC("bottom right corner", data, mesh)
    tr_corner = NeumannBC("top right corner", data, mesh)

    K, K_stored, R = vect_assembly(data, mesh, left_side, br_corner)
    
    tr_corner.impose(R)

    D = linalg.spsolve(K, R)

    D_true = np.array([
        0.0,  0.0,
        1.90773874e-05,  0.0,
        8.73032981e-06, -7.41539125e-05,
        0.0,  0.0
    ])
    np.testing.assert_allclose(D_true, D)