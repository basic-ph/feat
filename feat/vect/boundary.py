import numpy as np
from scipy import sparse

from ..base.boundary import build_dirichlet_data


def apply_dirichlet(nodes, K, R, *conditions):
    dir_dof, dir_values = build_dirichlet_data(*conditions)
    R[dir_dof] = dir_values
    # mask of booleans checking if (row) indices are present in dirichlet dof array
    mask_csc = np.in1d(K.indices, dir_dof)
    K.data[mask_csc] = 0.0  # elements are cleared directly from data sparse attribute
    K[dir_dof, dir_dof] = 1.0  # fancy indexing

    return K, R
