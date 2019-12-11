import numpy as np
from scipy import sparse

from feat import base, vect


A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])
A = sparse.csc_matrix(A)
print(A.toarray())
print()
print(A.indices)
print()
print(A.data)

dir_dof = np.array([1])
mask = np.in1d(A.indices, dir_dof)
print()
print(mask)

A.data[mask] = 0.0
A[dir_dof, dir_dof] = 1.0

print()
print(A.toarray())