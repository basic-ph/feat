"""
Testing Displacements solution against an example
"""

import numpy as np
from feat.fem import analysis


def test_analysis():
    data_path = "data/test.json"
    mesh_path = "gmsh/msh/test.msh"

    D = analysis(data_path, mesh_path)
    D_true = np.array([
        0.0,  0.0,
        1.90773874e-05,  0.0,
        8.73032981e-06, -7.41539125e-05,
        0.0,  0.0
    ])
    np.testing.assert_allclose(D_true, D)