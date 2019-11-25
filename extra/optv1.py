def assembly_opt_v1(e, data, mesh, E_matrices, K_flat, I, J):

    elements_num = mesh.cells["triangle"].shape[0]
    
    # K_loc: array with values from local stiffness matrix (column-wise)
    k = stiffness_matrix(e, data, mesh, E_matrices)
    K_loc = np.ravel(k, order="F")

    # I_loc: global row indices -- J_loc: global column indices
    element_dof = compute_global_dof(e, mesh)
    
    I_loc = np.tile(element_dof, 6)  # reps is the number of dof in each element
    J_loc = np.repeat(element_dof, 6)  # repeats is again the number of dof ^^
    start = 36 * e
    end = 36 * (e + 1)

    K_flat[start:end] = K_loc
    I[start:end] = I_loc
    J[start:end] = J_loc
    
    return K_flat, I, J


def test_assembly_opt_v1(setup_data, setup_mesh):
    data = setup_data("data/test.json")
    weights, locations = gauss_quadrature(data)
    mesh = setup_mesh("gmsh/msh/test.msh")
    elements = mesh.cells["triangle"].shape[0]
    nodes = mesh.points.shape[0]

    E_matrices = compute_E_matrices(data, mesh)
    K_flat = np.zeros(36 * elements)  # 36 is 6^2 (dofs^2)
    I = np.zeros(36 * elements, dtype=np.int32)  # the 2nd quantity is the number of elements
    J = np.zeros(36 * elements, dtype=np.int32)
    # testing only with element 1 (the 2nd)
    for e in range(elements):  # number of elements
        K_flat, I, J = assembly_opt_v1(e, data, mesh, E_matrices, K_flat, I, J)

    I_true = np.array([
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 4, 5, 6, 7,
        0, 1, 4, 5, 6, 7,
        0, 1, 4, 5, 6, 7,
        0, 1, 4, 5, 6, 7,
        0, 1, 4, 5, 6, 7,
        0, 1, 4, 5, 6, 7,
    ], dtype=np.int32)
    J_true = np.array([
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5,
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1,
        4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7,
    ], dtype=np.int32)
    
    np.testing.assert_allclose(I_true, I)
    np.testing.assert_allclose(J_true, J)