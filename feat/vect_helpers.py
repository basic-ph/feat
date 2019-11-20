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
    print(e)
    print(c)
    # print(E_array)
    test = y(c, e, 1, 2)
    test_ = x(c, e, 2, 1)
    print(test)
    print(test_)

    J = x(c,e,1,0) * y(c,e,2,0) - x(c,e,2,0) * y(c,e,1,0)

    K11 = ((y(c,e,1,2)**2 * E_array[:,0]) / (J**2) + (x(c,e,2,1)**2 * E_array[:,5]) / (J**2)) * t * 0.5 * J
    K12 = (y(c,e,1,2) * x(c,e,2,1) / J**2) * (E_array[:,1] + E_array[:,5]) * t * 0.5 * J
    K13 = (y(c,e,1,2) * y(c,e,2,0) * E_array[:,0] + x(c,e,2,1) * x(c,e,0,2) * E_array[:,5]) / (J**2) * t * 0.5 * J
    K14 = (y(c,e,1,2) * x(c,e,0,2) * E_array[:,1] + y(c,e,2,0) * x(c,e,2,1) * E_array[:,5]) / (J**2) * t * 0.5 * J
    K15 = (y(c,e,1,2) * y(c,e,0,1) * E_array[:,0] + x(c,e,2,1) * x(c,e,1,0) * E_array[:,5]) / (J**2) * t * 0.5 * J
    K16 = (y(c,e,1,2) * x(c,e,1,0) * E_array[:,1] + x(c,e,2,1) * y(c,e,0,1) * E_array[:,5]) / (J**2) * t * 0.5 * J
    
    K22 = (x(c,e,2,1)**2 * E_array[:,3] + y(c,e,1,2)**2 * E_array[:,5]) / (J**2) * t * 0.5 * J
    K23 = (x(c,e,2,1) * y(c,e,2,0) * E_array[:,1] + y(c,e,1,2) * x(c,e,0,2) * E_array[:,5]) / (J**2) * t * 0.5 * J
    K24 = (x(c,e,2,1) * x(c,e,0,2) * E_array[:,3] + y(c,e,1,2) * y(c,e,2,0) * E_array[:,5]) / (J**2) * t * 0.5 * J
    K25 = (x(c,e,2,1) * y(c,e,0,1) * E_array[:,1] + y(c,e,1,2) * x(c,e,1,0) * E_array[:,5]) / (J**2) * t * 0.5 * J
    K26 = (x(c,e,2,1) * x(c,e,1,0) * E_array[:,3] + y(c,e,1,2) * y(c,e,0,1) * E_array[:,5]) / (J**2) * t * 0.5 * J
    
    K33 = (y(c,e,2,0)**2 * E_array[:,0] + x(c,e,0,2)**2 * E_array[:,5]) / (J**2) * t * 0.5 * J
    K34 = (y(c,e,2,0) * x(c,e,0,2) * E_array[:,1] + x(c,e,0,2) * y(c,e,2,0) * E_array[:,5]) / (J**2) * t * 0.5 * J
    K35 = (y(c,e,2,0) * y(c,e,0,1) * E_array[:,0] + x(c,e,0,2) * x(c,e,1,0) * E_array[:,5]) / (J**2) * t * 0.5 * J
    K36 = (y(c,e,2,0) * x(c,e,1,0) * E_array[:,1] + x(c,e,0,2) * y(c,e,0,1) * E_array[:,5]) / (J**2) * t * 0.5 * J
    
    K44 = (x(c,e,0,2)**2 * E_array[:,3] + y(c,e,2,0)**2 * E_array[:,5]) / (J**2) * t * 0.5 * J
    K45 = (x(c,e,0,2) * y(c,e,0,1) * E_array[:,1] + y(c,e,2,0) * x(c,e,1,0) * E_array[:,5]) / (J**2) * t * 0.5 * J
    K46 = (x(c,e,0,2) * x(c,e,1,0) * E_array[:,3] + y(c,e,2,0) * y(c,e,0,1) * E_array[:,5]) / (J**2) * t * 0.5 * J
    
    K55 = (y(c,e,0,1)**2 * E_array[:,0] + x(c,e,1,0)**2 * E_array[:,5]) / (J**2) * t * 0.5 * J
    K56 = (y(c,e,0,1) * x(c,e,1,0) * E_array[:,1] + x(c,e,1,0) * y(c,e,0,1) * E_array[:,5]) / (J**2) * t * 0.5 * J
    
    K66 = (x(c,e,1,0)**2 * E_array[:,3] + y(c,e,0,1)**2 * E_array[:,5]) / (J**2) * t * 0.5 * J

    K_array = np.zeros((36, elements_num))
    K_array = np.stack((
        K11, K12, K13, K14, K15, K16,
        K12, K22, K23, K24, K25, K26,
        K13, K23, K33, K34, K35, K36,
        K14, K24, K34, K44, K45, K46,
        K15, K25, K35, K45, K55, K56,
        K16, K26, K36, K46, K56, K66,
    ))

    return K_array


def vect_compute_global_dof(mesh):
    nodes = mesh.points.shape[0]
    elements = mesh.cells["triangle"]
    elements_dof = np.zeros((elements.shape[0], 6), dtype=np.int32)
    for n in range(3):  # 3 is the number of nodes
        elements_dof[:, n*2] = elements[:, n] * 2
        elements_dof[:, n*2+1] = elements[:, n] * 2 + 1
    return elements_dof