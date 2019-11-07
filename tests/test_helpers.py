import numpy as np
import pytest

from feat.helpers import compute_E_matrices, stiffness_matrix, x, y

base_dir = "/home/basic-ph/thesis/feat/"

def test_compute_E_matrices(setup_data, setup_mesh):
    data = setup_data(base_dir + "data/test_el_0.json")
    mesh = setup_mesh(base_dir + "gmsh/msh/test_el_0.msh")

    E_matrices = compute_E_matrices(data, mesh)
    E_true = np.array([
        (3.2e7, 8e6, 0.0),
        (8e6, 3.2e7, 0.0),
        (0.0, 0.0, 1.2e7),
    ])

    assert E_matrices[4]["name"] == "steel"
    np.testing.assert_allclose(E_true, E_matrices[4]["E"])


k_true = np.array([
    (9833333.33333333, -0.5e7, -0.45e7, 0.2e7, -5333333.33333333, 0.3e7),
    (-0.5e7, 1.4e7, 0.3e7, -1.2e7, 0.2e7, -0.2e7),
    (-0.45e7, 0.3e7, 0.45e7, 0.0, 0.0, -0.3e7),
    (0.2e7, -1.2e7, 0.0, 1.2e7, -0.2e7, 0.0),
    (-5333333.33333333, 0.2e7, 0.0, -0.2e7, 5333333.33333333, 0.0),
    (0.3e7, -0.2e7, -0.3e7, 0.0, 0.0, 0.2e7),
])
test_data = [
    (
        "data/test_el_0.json",
        "gmsh/msh/test_el_0.msh",
        k_true,
    ),
    (
        "data/test_el_1.json",
        "gmsh/msh/test_el_1.msh",
        k_true,
    )
]


@pytest.mark.parametrize("data_file,mesh_file,k_true", test_data)
def test_stiffness_matrix(setup_data, setup_mesh, data_file, mesh_file, k_true):
    data = setup_data(base_dir + data_file)
    mesh = setup_mesh(base_dir + mesh_file)
    nodal_coordinates = mesh.points[:,:2]
    connectivity_table = mesh.cells["triangle"]
    element_material_map = mesh.cell_data["triangle"]["gmsh:physical"]
    E_matrices = compute_E_matrices(data, mesh)

    k = stiffness_matrix(
        0,
        data,
        mesh,
        nodal_coordinates,
        connectivity_table,
        element_material_map,
        E_matrices
    )
    np.testing.assert_allclose(k_true, k)
