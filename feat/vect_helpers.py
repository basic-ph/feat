import numpy as np
from scipy import sparse
from helpers import stiffness_matrix, compute_global_dof
from boundary import dirichlet_dof


def vect_compute_E(data, mesh, elements_num):
    condition = data["load condition"]
    materials_num = len(data["materials"].keys())

    E_mat = np.zeros((materials_num, 6)) # pre-computed array for each material

    for key,value in data["materials"].items():
        # key is the material name
        # value is the dict with young's modulus and poisson's ratio
        physical_tag = mesh.field_data[key][0]
        
        poisson = value["poisson's ratio"]
        young = value["young's modulus"]

        if condition == "plane strain":
            E = np.array([
                young * (1 - poisson) / ((1 + poisson) * (1 - 2*poisson)),
                young * poisson / ((1 + poisson) * (1 - 2*poisson)),
                0.0,
                young * (1 - poisson) / ((1 + poisson) * (1 - 2*poisson)),
                0.0,
                young * (1 - poisson) / (2 * (1 + poisson) * (1 - 2*poisson)),
            ])
        elif condition == "plane stress":
            E = np.array([
                young / (1 - poisson**2),
                young * poisson / (1 - poisson**2),
                0.0,
                young / (1 - poisson**2),
                0.0,
                young * (1 - poisson) / (2 * (1 - poisson**2)),
            ])
        E_mat[physical_tag-1,:] = E
    
    E_array = np.zeros((elements_num, 6))
    mat_map = mesh.cell_data["triangle"]["gmsh:physical"] - 1  # element-material map
    E_array = E_mat[mat_map,:]

    return E_array


def vect_compute_K_entry(row, col, c, e, E_array, t):

    J = ((c[e[:,1]][:,0] - c[e[:,0]][:,0]) * (c[e[:,2]][:,1] - c[e[:,0]][:,1]) -
        (c[e[:,2]][:,0] - c[e[:,0]][:,0]) * (c[e[:,1]][:,1] - c[e[:,0]][:,1]))
    
    b = np.array([
        (c[e[:,1]][:,1] - c[e[:,2]][:,1], c[e[:,2]][:,0] - c[e[:,1]][:,0]),
        (c[e[:,2]][:,0] - c[e[:,1]][:,0], c[e[:,1]][:,1] - c[e[:,2]][:,1]),
        (c[e[:,2]][:,1] - c[e[:,0]][:,1], c[e[:,0]][:,0] - c[e[:,2]][:,0]),
        (c[e[:,0]][:,0] - c[e[:,2]][:,0], c[e[:,2]][:,1] - c[e[:,0]][:,1]),
        (c[e[:,0]][:,1] - c[e[:,1]][:,1], c[e[:,1]][:,0] - c[e[:,0]][:,0]),
        (c[e[:,1]][:,0] - c[e[:,0]][:,0], c[e[:,0]][:,1] - c[e[:,1]][:,1]),
    ])
    
    E_indices = np.array([(0, 1, 0, 1, 0, 1), (1, 3, 1, 3, 1, 3)])

    if (row % 2 == 0):
        E = E_array[:, E_indices[0, col]]
    else:
        E = E_array[:, E_indices[1, col]]

    k_data = (b[row][0] * b[col][0] * E + b[row][1] * b[col][1] * E_array[:,5]) / (J**2) * t * 0.5 * J
    return k_data


def vect_compute_global_dof(mesh, row, col):
    elements = mesh.cells["triangle"]
    elements_num = mesh.cells["triangle"].shape[0]
    row_ind = np.zeros((elements_num))
    col_ind = np.zeros((elements_num))

    if (row % 2 == 0):
        row_ind = elements[:, row // 2] * 2
    else:
        row_ind = elements[:, row // 2] * 2 + 1
    
    if (col % 2 == 0):
        col_ind = elements[:, col // 2] * 2
    else:
        col_ind = elements[:, col // 2] * 2 + 1

    return row_ind, col_ind
    


def vect_assembly(data, mesh, *conditions):

    t = data["thickness"]
    nodes = mesh.points.shape[0]
    elements_num = mesh.cells["triangle"].shape[0]
    elements = mesh.cells["triangle"]  # elements mapping, n-th row: nodes in n-th element
    coord = mesh.points[:,:2]  # x, y coordinates

    k_data = np.zeros((elements_num))
    row_ind = np.zeros((elements_num))
    col_ind = np.zeros((elements_num))
    r_data = np.zeros((elements_num))

    E_array = vect_compute_E(data, mesh, elements_num)

    K = sparse.csc_matrix((2 * nodes, 2 * nodes))
    R = np.zeros((2 * nodes))
    
    for (row, col) in zip(*np.triu_indices(6, k=1)):  # upper triangular data
        k_data = vect_compute_K_entry(row, col, coord, elements, E_array, t)
        row_ind, col_ind = vect_compute_global_dof(mesh, row, col)

        for c in conditions:
            # creating masks (boolean arrays) used to access contrained data in k_data
            row_mask = np.isin(row_ind, c.global_dof)  # check if each element in row_ind is part of gloabal_dof (True) or not (False)
            col_mask = np.isin(col_ind, c.global_dof)
            r_data[col_mask] -= k_data[col_mask] * c.imposed_disp  # move contrained columns data to rhs data
            k_data[row_mask] = 0.0  # zero-out using row_mask
            k_data[col_mask] = 0.0  # zero-out using col_mask

        K += sparse.csc_matrix((k_data, (row_ind, col_ind)),shape=(8,8))
        R[row_ind] += r_data
    
    K = K + K.transpose()

    for (row, col) in zip(*np.diag_indices(6)):  # diagonal data
        k_data = vect_compute_K_entry(row, col, coord, elements, E_array, t)
        row_ind, col_ind = vect_compute_global_dof(mesh, row, col)

        for c in conditions:
            #
            col_mask = np.isin(col_ind, c.global_dof)
            r_data[col_mask] -= k_data[col_mask] * c.imposed_disp  # move contrained columns data to rhs data
            k_data[col_mask] = 1.0  # zero-out using col_mask FIXME decide if use avg. value for condition number improvement

        K += sparse.csc_matrix((k_data, (row_ind, col_ind)),shape=(8,8))
        R[row_ind] += r_data
    print(K.toarray())
    print()
    return K, R