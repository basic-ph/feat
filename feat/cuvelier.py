
# 1st method (direct solve)
# ID: index array of dirichlet boundary nodes (dof or nodes??)
# IDc: its complementary (all other not constrained dofs??)
# gD: extension to zero of dirichlet boundary function (??)
#     maybe this is the array of bc values for each DOF in the problem with zeros for dofs without constraints
# A,b: matrix and vector of the system without taking into account dirichlet bc

# x 1d array with dim equal to number of dof
x = np.zeros(ndof)
# select element of gD related to dirichlet boundary nodes
x[ID] = gD[ID]  
# move to rhs the data from A matrix related to not constrained dof and multiply for the bc value
bb = b[IDc] - A[IDc,::] * gD  
# solve the system but first indexing A matrix and x vector
# A[IDc] select the not constrained rows, then [::,IDc] picks all rows of not constrained columns
# in other words it is taking only not constrained entries from main matrix
x[IDc] = sparse.spsolve((A[IDc])[::,IDc], bb)


# 2nd method (set dirichlet rows to identity without changing the sparsity structure)

# rows_to_identity maybe are the indices of Dirichlet rows that have to become identity row
# create a mask of booleans checking if indices (row indices in a csc sparse matrix) are present in rows_to_identity
mask = np.in1d(A.indices, rows_to_identity)
# zero out values inside A (contained in attribute data) using mask for indexing all the entries that needs to be cleared
# (the concept of row in this format is no more useful cause indices and data are in one single flat array)
A.data[mask] = 0.0
# set all diagonal entries related to Dirichled bc to 1
A[rows_to_identity, rows_to_identity] = 1.0

# to use this with inhomogeneous bc we need to take care of moving contributions to rhs
# like in my old code... column slicing and moving to rhs its ok with csc matrix