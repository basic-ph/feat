import numpy as np

from feat.tools import compute_E_matrices


def test_compute_E_matrices(setup_data, setup_mesh):
    data = setup_data(r"data\test_tools.json")
    mesh = setup_mesh(r"gmsh\msh\2d_bar.msh")

    E_matrices = compute_E_matrices(data, mesh)
    assert E_matrices[3]["name"] == "dummy"
    np.testing.assert_allclose(
        E_matrices[4]["E"],
        np.array([
            (269.230769, 115.384615, 0.0),
            (115.384615, 269.230769, 0.0),
            (0.0, 0.0, 76.923077),
        ])
    )