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


x = lambda a, i, j: a[i][0] - a[j][0]
y = lambda b, i, j: b[i][1] - b[j][1]



def vect_stiffness_matrix(data, mesh, E_array):
    
    t = data["thickness"]
    elements = mesh.cells["triangle"]
    c = mesh.points[:,:2]  # x, y coordinates
    print(elements)
    print(c)
    print(E_array)

    print(c[elements[:,1]])
    return 0


def vect_compute_global_dof(mesh):
    nodes = mesh.points.shape[0]
    elements = mesh.cells["triangle"]
    elements_dof = np.zeros((elements.shape[0], 6), dtype=np.int32)
    for n in range(3):  # 3 is the number of nodes
        elements_dof[:, n*2] = elements[:, n] * 2
        elements_dof[:, n*2+1] = elements[:, n] * 2 + 1
    return elements_dof