import meshio


mesh = meshio.read("../gmsh/msh/test.msh")
print("mesh.points:\n", mesh.points)
print()
print("mesh.cells:\n", mesh.cells)
print()
print("mesh.cell_data:", mesh.cell_data)
print()
print("mesh.field_data: ", mesh.field_data)

# mesh.points:
#  [[0.  0.  0. ]
#  [3.  0.  0. ]
#  [3.  2.  0. ]
#  [0.  2.  0. ]
#  [1.5 1.  0. ]]

# mesh.cells:
#  {'vertex': array([[1], [2]]), 
#   'line': array([[3, 0]]),
#   'triangle': array(
#       [[3, 0, 4],
#        [1, 2, 4],
#        [0, 1, 4],
#        [2, 3, 4]])}

# mesh.cell_data: {
# 'vertex': {'gmsh:physical': array([7, 8])},
# 'line': {'gmsh:physical': array([6])},
# 'triangle': {'gmsh:physical': array([5, 5, 5, 5])}}

# mesh.field_data:  {
# 'bottom right corner': array([7, 0]),
# 'top right corner': array([8, 0]),
# 'left side': array([6, 1]),
# 'steel': array([5, 2])}
