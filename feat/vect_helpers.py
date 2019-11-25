import numpy as np
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


def vect_compute_global_dof(mesh):
    nodes = mesh.points.shape[0]
    elements = mesh.cells["triangle"]
    elements_dof = np.zeros((elements.shape[0], 6), dtype=np.int32)
    for n in range(3):  # 3 is the number of nodes
        elements_dof[:, n*2] = elements[:, n] * 2
        elements_dof[:, n*2+1] = elements[:, n] * 2 + 1
    return elements_dof


def vect_assembly(data, mesh, E_array):

    t = data["thickness"]
    elements_num = mesh.cells["triangle"].shape[0]
    e = mesh.cells["triangle"]  # elements mapping, n-th row: nodes in n-th element
    c = mesh.points[:,:2]  # x, y coordinates
    K_array = np.zeros((36, elements_num))
    I_array = np.zeros((36, elements_num))
    J_array = np.zeros((36, elements_num))

    elements_dof = vect_compute_global_dof(mesh)
    print(elements_dof)
    print()

    indip_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 14, 15, 16, 17, 21, 22, 23, 28, 29, 35]
    tril_indices = [6, 12, 13, 18, 19, 20, 24, 25, 26, 27, 30, 31, 32, 33, 34]  # tril: lower-triangle array
    triu_indices = [1,  2,  8,  3,  9, 15,  4, 10, 16, 22,  5, 11, 17, 23, 29]  # triu: upper-triangle array

    for l in indip_indices:
        row, col = np.unravel_index(l, (6,6))
        K_array[l] = vect_compute_K_entry(row, col, c, e, E_array, t)
        I_array[l] = elements_dof[:, row]
        J_array[l] = elements_dof[:, col]

    K_array[tril_indices] = K_array[triu_indices]
    I_array[tril_indices] = J_array[triu_indices]
    J_array[tril_indices] = I_array[triu_indices]

    return K_array, I_array, J_array