import numpy as np
from scipy import sparse

from feat import base, vect


A = np.array([
    [2, 2, 2],
    [5, 5, 5],
    [8, 8, 8],
])
A = sparse.csc_matrix(A)

print("original matrix:\n", A.toarray())
print()
print("matrix indices:\n", A.indices)
print()
print("matrix data:\n", A.data)
print()

dir_dof = np.array([1])
mask = np.in1d(A.indices, dir_dof)

print("mask:\n", mask)
print()

A.data[mask] = 0.0
A[dir_dof, dir_dof] = 1.0

print("modified matrix:\n", A.toarray())
print()