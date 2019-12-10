import numpy as np
from scipy import sparse

def compute_E_array(mesh, *materials):
    elements_num = mesh.cells["triangle"].shape[0]
    materials_num = len(materials)
    E_array = np.zeros((elements_num, 6))
    E_material = np.zeros((materials_num, 6)) # pre-computed array for each material
    material_map = mesh.cell_data["triangle"]["gmsh:physical"] - 1  # element-material map

    for m in materials:
        # print(m)
        E_material[m.tag] = m.E_flat
    
    E_array = E_material[material_map]
    return E_array


def compute_K_entry(row, col, c, e, E_array, t):

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


def compute_global_dof(mesh, row, col):
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
    

def assembly(mesh, E_array, thickness):

    t = thickness
    nodes = mesh.points.shape[0]
    elements_num = mesh.cells["triangle"].shape[0]
    elements = mesh.cells["triangle"]  # elements mapping, n-th row: nodes in n-th element
    coord = mesh.points[:,:2]  # x, y coordinates

    k_data = np.zeros((elements_num))
    row_ind = np.zeros((elements_num))
    col_ind = np.zeros((elements_num))
    K = sparse.csc_matrix((2 * nodes, 2 * nodes))
    
    for (row, col) in zip(*np.triu_indices(6, k=1)):
        k_data = compute_K_entry(row, col, coord, elements, E_array, t)
        row_ind, col_ind = compute_global_dof(mesh, row, col)
        K += sparse.csc_matrix((k_data, (row_ind, col_ind)),shape=(8,8))
    
    K = K + K.transpose()

    for (row, col) in zip(*np.diag_indices(6)):
        k_data = compute_K_entry(row, col, coord, elements, E_array, t)
        row_ind, col_ind = compute_global_dof(mesh, row, col)
        K += sparse.csc_matrix((k_data, (row_ind, col_ind)),shape=(8,8))

    return K
