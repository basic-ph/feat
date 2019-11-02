import json

import meshio
import numpy as np

mesh = meshio.read(r"..\gmsh\msh\2d_bar.msh")
print("mesh.points:\n", mesh.points)
print()
print("mesh.cells:\n", mesh.cells)
print()
print("mesh.cell_data:", mesh.cell_data)
print()
print("mesh.field_data: ", mesh.field_data)

# mesh.points:
#  [[0.  0.  0. ]
#  [1.  0.  0. ]
#  [2.  0.  0. ]
#  [2.  1.  0. ]
#  [1.  1.  0. ]
#  [0.  1.  0. ]
#  [0.5 0.5 0. ]
#  [1.5 0.5 0. ]]

# mesh.cells:
#  {'line': array([[2, 3],
#        [5, 0]]), 'triangle': array([[0, 1, 6],
#        [5, 0, 6],
#        [1, 4, 6],
#        [4, 5, 6],
#        [1, 2, 7],
#        [4, 1, 7],
#        [2, 3, 7],
#        [3, 4, 7]])}

# mesh.cell_data: {'line': {'gmsh:physical': array([2, 1])}, 'triangle': {'gmsh:physical': array([3, 3, 3, 3, 4, 4, 4, 4])}}

# mesh.field_data:  {'left side': array([1, 1]), 'right side': array([2, 1]), 'dummy': array([3, 2]), 'steel': array([4, 2])}