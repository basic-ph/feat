import numpy as np
from scipy import sparse


def build_dirichlet_data(nodes, *conditions):
    dir_dof_list = [c.global_dof for c in conditions]
    dir_values_list = [c.values for c in conditions]
    dir_dof = np.concatenate(dir_dof_list)
    dir_values = np.concatenate(dir_values_list)

    # extension to zero of dirichlet values, each dof in mesh is present
    # also dof that do not belong to any Dirichlet boundary condition
    ext_dir_values = np.zeros(nodes * 2)
    ext_dir_values[dir_dof] = dir_values

    return dir_dof, ext_dir_values


def apply_dirichlet(nodes, K, R, *conditions):
    dir_dof, ext_dir_values = build_dirichlet_data(nodes, *conditions)
    print("dir_dof", dir_dof)
    print("ext_dir_values", ext_dir_values)

    K = K.tocsr()  # faster matrix-vector product
    R -= K.dot(ext_dir_values)  # equivalent operation of column-wise subtraction
    
    mask_csr = np.in1d(K.indices, dir_dof)
    K.data[mask_csr] = 0.0

    K = K.tocsc()  # FIXME are these conversions ok?
    # mask of booleans checking if (row) indices are present in dirichlet dof array
    mask_csc = np.in1d(K.indices, dir_dof)
    K.data[mask_csc] = 0.0  # elements are cleared directly from data sparse attribute
    K[dir_dof, dir_dof] = 1.0  # fancy indexing
    # print(K.toarray())
    # print()

    return K, R
