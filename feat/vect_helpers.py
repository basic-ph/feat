import numpy as np
from scipy import sparse
from helpers import stiffness_matrix, compute_global_dof


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

    k = (b[row][0] * b[col][0] * E + b[row][1] * b[col][1] * E_array[:,5]) / (J**2) * t * 0.5 * J
    return k


def vect_compute_global_dof(mesh, row, col):
    elements = mesh.cells["triangle"]
    elements_num = mesh.cells["triangle"].shape[0]
    I_indices = np.zeros((elements_num))
    J_indices = np.zeros((elements_num))

    if (row % 2 == 0):
        I_indices = elements[:, row // 2] * 2
    else:
        I_indices = elements[:, row // 2] * 2 + 1
    
    if (col % 2 == 0):
        J_indices = elements[:, col // 2] * 2
    else:
        J_indices = elements[:, col // 2] * 2 + 1

    return I_indices, J_indices
    


def vect_assembly(data, mesh):

    t = data["thickness"]
    nodes = mesh.points.shape[0]
    elements_num = mesh.cells["triangle"].shape[0]
    e = mesh.cells["triangle"]  # elements mapping, n-th row: nodes in n-th element
    c = mesh.points[:,:2]  # x, y coordinates

    E_array = vect_compute_E(data, mesh, elements_num)

    indip_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 14, 15, 16, 17, 21, 22, 23, 28, 29, 35]
    tril_indices = [6, 12, 13, 18, 19, 20, 24, 25, 26, 27, 30, 31, 32, 33, 34]  # tril: lower-triangle array
    triu_indices = [1,  2,  8,  3,  9, 15,  4, 10, 16, 22,  5, 11, 17, 23, 29]  # triu: upper-triangle array

    K = sparse.csc_matrix((2 * nodes, 2 * nodes))
    
    for l in indip_indices:
        row, col = np.unravel_index(l, (6,6))
        K_entries = vect_compute_K_entry(row, col, c, e, E_array, t)
        I_indices, J_indices = vect_compute_global_dof(mesh, row, col)
        K += sparse.csc_matrix((K_entries, (I_indices, J_indices)),shape=(8,8))
    for l in tril_indices:
        row, col = np.unravel_index(l, (6,6))
        K_entries = vect_compute_K_entry(row, col, c, e, E_array, t)
        I_indices, J_indices = vect_compute_global_dof(mesh, row, col)
        K += sparse.csc_matrix((K_entries, (I_indices, J_indices)),shape=(8,8))

    return K
