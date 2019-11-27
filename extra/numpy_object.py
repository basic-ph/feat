  nimport numpy as np
from scipy import sparse

R = np.zeros(8)
imposed = 0.0
row  = np.array([0, 3, 1, 0])
col  = np.array([0, 0, 1, 2])
data = np.array([4., 5., 7., 9.])
K = sparse.coo_matrix((data, (row, col)))
K = K.tolil()

print(R)
print(K.toarray())
print(K.dtype)

column = K[:,0].toarray()  # workaround
flat_column = np.ravel(column)
# column = column.astype(float)Z

print(column)
print(column.dtype)
print()
print(flat_column)
print(flat_column.dtype)

