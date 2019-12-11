import meshio

mesh = meshio.read("./data/msh/base.msh")
print("mesh.points:\n", mesh.points)
print()
print("mesh.cells:\n", mesh.cells)
print()
print("mesh.cell_data:", mesh.cell_data)
print()
print("mesh.field_data: ", mesh.field_data)

# mesh.points:
#  [[0. 0. 0.]
#  [2. 0. 0.]
#  [2. 2. 0.]
#  [0. 2. 0.]
#  [1. 1. 0.]]

# mesh.cells:
#  {'vertex': array([[0]]),
#   'line': array([
#       [1, 2],
#       [3, 0]
#    ]), 
#   'triangle': array([
#        [0, 1, 4],
#        [3, 0, 4],
#        [1, 2, 4],
#        [2, 3, 4]
#   ])}

# mesh.cell_data: {'vertex': {'gmsh:physical': array([3])}, 'line': {'gmsh:physical': array([4, 2])}, 'triangle': {'gmsh:physical': array([1, 1, 1, 1])}}

# mesh.field_data:  {'bottom left corner': array([3, 0]), 'left side': array([2, 1]), 'right side': array([4, 1]), 'cheese': array([1, 2])}