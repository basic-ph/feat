import numpy as np
from helpers import stiffness_matrix, compute_global_dof


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


x = lambda c, e, i, j: c[e[:,i]][:,0] - c[e[:,j]][:,0]
y = lambda c, e, i, j: c[e[:,i]][:,1] - c[e[:,j]][:,1]


def vect_stiffness_matrix(data, mesh, E_array):
    
    t = data["thickness"]
    elements_num = mesh.cells["triangle"].shape[0]
    e = mesh.cells["triangle"]  # elements mapping, n-th row: nodes in n-th element
    c = mesh.points[:,:2]  # x, y coordinates
    K_array = np.zeros((36, elements_num))

    J = x(c,e,1,0) * y(c,e,2,0) - x(c,e,2,0) * y(c,e,1,0)

    K_array[0] = ((y(c,e,1,2)**2 * E_array[:,0]) / (J**2) + (x(c,e,2,1)**2 * E_array[:,5]) / (J**2)) * t * 0.5 * J  # K11
    K_array[1] = (y(c,e,1,2) * x(c,e,2,1) / J**2) * (E_array[:,1] + E_array[:,5]) * t * 0.5 * J  # K12
    K_array[2] = (y(c,e,1,2) * y(c,e,2,0) * E_array[:,0] + x(c,e,2,1) * x(c,e,0,2) * E_array[:,5]) / (J**2) * t * 0.5 * J  # K13
    K_array[3] = (y(c,e,1,2) * x(c,e,0,2) * E_array[:,1] + y(c,e,2,0) * x(c,e,2,1) * E_array[:,5]) / (J**2) * t * 0.5 * J  # K14
    K_array[4] = (y(c,e,1,2) * y(c,e,0,1) * E_array[:,0] + x(c,e,2,1) * x(c,e,1,0) * E_array[:,5]) / (J**2) * t * 0.5 * J  # K15
    K_array[5] = (y(c,e,1,2) * x(c,e,1,0) * E_array[:,1] + x(c,e,2,1) * y(c,e,0,1) * E_array[:,5]) / (J**2) * t * 0.5 * J  # K16
    
    K_array[7] = (x(c,e,2,1)**2 * E_array[:,3] + y(c,e,1,2)**2 * E_array[:,5]) / (J**2) * t * 0.5 * J  # K22
    K_array[8] = (x(c,e,2,1) * y(c,e,2,0) * E_array[:,1] + y(c,e,1,2) * x(c,e,0,2) * E_array[:,5]) / (J**2) * t * 0.5 * J  # K23
    K_array[9] = (x(c,e,2,1) * x(c,e,0,2) * E_array[:,3] + y(c,e,1,2) * y(c,e,2,0) * E_array[:,5]) / (J**2) * t * 0.5 * J  # K24
    K_array[10] = (x(c,e,2,1) * y(c,e,0,1) * E_array[:,1] + y(c,e,1,2) * x(c,e,1,0) * E_array[:,5]) / (J**2) * t * 0.5 * J  # K25
    K_array[11] = (x(c,e,2,1) * x(c,e,1,0) * E_array[:,3] + y(c,e,1,2) * y(c,e,0,1) * E_array[:,5]) / (J**2) * t * 0.5 * J  # K26
    
    K_array[14] = (y(c,e,2,0)**2 * E_array[:,0] + x(c,e,0,2)**2 * E_array[:,5]) / (J**2) * t * 0.5 * J  # K33
    K_array[15] = (y(c,e,2,0) * x(c,e,0,2) * E_array[:,1] + x(c,e,0,2) * y(c,e,2,0) * E_array[:,5]) / (J**2) * t * 0.5 * J  # K34
    K_array[16] = (y(c,e,2,0) * y(c,e,0,1) * E_array[:,0] + x(c,e,0,2) * x(c,e,1,0) * E_array[:,5]) / (J**2) * t * 0.5 * J  # K35
    K_array[17] = (y(c,e,2,0) * x(c,e,1,0) * E_array[:,1] + x(c,e,0,2) * y(c,e,0,1) * E_array[:,5]) / (J**2) * t * 0.5 * J  # K36
    
    K_array[21] = (x(c,e,0,2)**2 * E_array[:,3] + y(c,e,2,0)**2 * E_array[:,5]) / (J**2) * t * 0.5 * J  # K44
    K_array[22] = (x(c,e,0,2) * y(c,e,0,1) * E_array[:,1] + y(c,e,2,0) * x(c,e,1,0) * E_array[:,5]) / (J**2) * t * 0.5 * J  # K45
    K_array[23] = (x(c,e,0,2) * x(c,e,1,0) * E_array[:,3] + y(c,e,2,0) * y(c,e,0,1) * E_array[:,5]) / (J**2) * t * 0.5 * J  # K46
    
    K_array[28] = (y(c,e,0,1)**2 * E_array[:,0] + x(c,e,1,0)**2 * E_array[:,5]) / (J**2) * t * 0.5 * J  # K55
    K_array[29] = (y(c,e,0,1) * x(c,e,1,0) * E_array[:,1] + x(c,e,1,0) * y(c,e,0,1) * E_array[:,5]) / (J**2) * t * 0.5 * J   # K56
    
    K_array[35] = (x(c,e,1,0)**2 * E_array[:,3] + y(c,e,0,1)**2 * E_array[:,5]) / (J**2) * t * 0.5 * J  # K11


    tril_indices = [6, 12, 13, 18, 19, 20, 24, 25, 26, 27, 30, 31, 32, 33, 34]  # tril: lower-triangle array
    triu_indices = [1,  2,  8,  3,  9, 15,  4, 10, 16, 22,  5, 11, 17, 23, 29]
    K_array[tril_indices] = K_array[triu_indices]

    return K_array


def compute_K_entry(l, data, mesh, E_array):

    t = data["thickness"]
    e = mesh.cells["triangle"]  # elements mapping, n-th row: nodes in n-th element
    c = mesh.points[:,:2]  # x, y coordinates

    J = x(c,e,1,0) * y(c,e,2,0) - x(c,e,2,0) * y(c,e,1,0)

    row, col = np.unravel_index(l, (6,6))

    b_0 = [y(c,e,1,2), x(c,e,2,1), y(c,e,2,0), x(c,e,0,2), y(c,e,0,1), x(c,e,1,0)]
    b_1 = [x(c,e,2,1), y(c,e,1,2), x(c,e,0,2), y(c,e,2,0), x(c,e,1,0), y(c,e,0,1)]
    E_indices = np.array([(0, 1, 0, 1, 0, 1), (1, 3, 1, 3, 1, 3)])
    
    if (row % 2 == 0):
        print(f"{row} is even")
        E = E_array[:, E_indices[0, col]]
    else:
        print(f"{row} is odd")
        E = E_array[:, E_indices[1, col]]

    k = (b_0[row] * b_0[col] * E + b_1[row] * b_1[col] * E_array[:,5]) / (J**2) * t * 0.5 * J
    return k


def vect_compute_global_dof(mesh):
    nodes = mesh.points.shape[0]
    elements = mesh.cells["triangle"]
    elements_dof = np.zeros((elements.shape[0], 6), dtype=np.int32)
    for n in range(3):  # 3 is the number of nodes
        elements_dof[:, n*2] = elements[:, n] * 2
        elements_dof[:, n*2+1] = elements[:, n] * 2 + 1
    return elements_dof