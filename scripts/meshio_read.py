"""
mesh.points: array dei nodi con loro coordinate
mesh.cells:dict con elementi divisi per tipo e array con nodi (zero-based) per ciascun elemento
mesh.cell_data: PhysicalTag degli elementi suddivisi per tipo
mesh.field_data: PhysicalNames mappa il nome, la PhysicalTag e la dimensione
"""

import meshio

mesh = meshio.read(r"./data/msh/feap.msh")
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
#  [0. 2. 0.]]

# mesh.cells:
#  {'vertex': array([[0],
#        [1],
#        [2],
#        [3]]), 'triangle': array([[0, 1, 2],
#        [0, 2, 3]])}

# mesh.cell_data: {'vertex': {'gmsh:physical': array([2, 4, 5, 3])}, 'triangle': {'gmsh:physical': array([1, 1])}}

# mesh.field_data:  {'bottom left corner': array([2, 0]), 'top left corner': array([3, 0]), 'bottom right corner': array([4, 0]), 'top right corner': array([5, 0]), 'cheese': array([1, 2])}