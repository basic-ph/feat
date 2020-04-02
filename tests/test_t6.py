import meshio
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg

from feat import base
from feat import boundary as bc
from feat import vector
from feat import T6

@pytest.mark.skip()
def test_stiffness_matrix():
    mesh_path = "tests/data/msh/felippa_t6.msh"

    element_type = "triangle6"
    integration_points = 3  # this must be 3 for T6
    load_condition = "plane stress"  # "plane stress" or "plane strain"
    thickness = 1.0
    berillium = base.Material("Be", 288, 0.333, load_condition)

    mesh = meshio.read(mesh_path)
    elements_num = mesh.cells_dict[element_type].shape[0]
    print("el num", elements_num)
    nodes = mesh.points.shape[0]
    E_material = base.compute_E_array(mesh, element_type, berillium)

    # k_0 = T6.stiffness_matrix(0, mesh, E_material, thickness, element_type, integration_points)
    k_0 = base.stiffness_matrix(0, mesh, E_material, thickness, element_type, integration_points)

    k_0_true = np.array([
        (5333333.33333333, 0.0, -5333333.33333333, 2000000., 0., -2000000.),
        (0., 2000000., 3000000., -2000000., -3000000., 0.),
        (-5333333.33333333, 3000000., 9833333.33333333, -5000000., -4500000., 2000000.),
        (2000000.,-2000000., -5000000., 14000000., 3000000., -12000000.),
        (0., -3000000., -4500000., 3000000., 4500000., 0.),
        (-2000000., 0., 2000000., -12000000., 0., 12000000.),
    ])
    
    np.set_printoptions(linewidth=200)
    
    print(k_0)
    # limit = 1e-12
    # over_limit = abs(k_0) < limit
    # print("over lim\n", over_limit)
    # k_0[over_limit] = 0.0
    # print(k_0)
    assert False
    # np.testing.assert_allclose(k_0_true, k_0)
