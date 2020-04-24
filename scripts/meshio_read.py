import meshio

mesh = meshio.read("../data/msh/base.msh")  # change me

print("mesh.points:\n", mesh.points)
# OUTPUT
# mesh.points:
#  [[0. 0. 0.]
#  [2. 0. 0.]
#  [2. 2. 0.]
#  [0. 2. 0.]
#  [1. 1. 0.]]
# node coordinates in 3D space as [x, y, z]

print("mesh.cells:\n", mesh.cells)
# OUTPUT
# mesh.cells: # 
#  [CellBlock(type='vertex', data=array([[0]])), 
#   CellBlock(type='line', data=array([[1, 2]])), 
#   CellBlock(type='line', data=array([[3, 0]])), 
#   CellBlock(type='triangle', data=array([  
#        [0, 1, 4], ---> [node1, node2, node3] forming the first 3-node triangle
#        [3, 0, 4],
#        [1, 2, 4],
#        [2, 3, 4]
#   ]))]
# list of namedtuples (https://docs.python.org/3/library/collections.html#collections.namedtuple)
# contains a CellBlock (structure containing "type" and "data" fields) for each gmsh physical group.
# Here two different cellblocks have the same "type" because there are two "line" physical groups (see below)
# "data" field contains the number (tag or label) of the nodes that constitute the element

print("mesh.point_data:\n", mesh.point_data)
# empty for this .geo file
# OUTPUT
# mesh.point_data:
#  {}

print("mesh.cell_data:\n", mesh.cell_data)
# OUTPUT
# mesh.cell_data:
#  {'gmsh:physical': [array([3]), array([4]), array([2]), array([1, 1, 1, 1])]}
# dictionary of lists: in this case the only key is "gmsh:physical" because the only 
# additional data gmsh provides regarding cells (elements) is which physical group they
# belong to. 
# In this case meshio is telling us that the only element inside the first CellBlock (vertex)
# belongs to physical group with tag = 3 ("bottom left corner"), also the four triangular elements inside the last
# CellBlock are all related to physical group with tag = 1 ("cheese")

print("mesh.field_data:\n", mesh.field_data)
# OUTPUT
# mesh.field_data:
#  {'bottom left corner': array([3, 0]), 'left side': array([2, 1]), 'right side': array([4, 1]), 'cheese': array([1, 2])}
# Dictionary containing detail regarding physical groups.
# Each key correspond to the physical group name we gave in the .geo file
# Each value is an array composed in this way: [<physical_tag>, <dimension>] in which <dimension> is 1, 2 or 3
# depending on whether the physical group is one-dimensional, two-dimensional or three-dimensional
# in this example "bottom left corner" has tag = 3 and dimension is zero 'cause it represents a single point
# "cheese" has tag = 1 and dimension = 2 because it includes the whole square surface.


# DEPRECATED attributes, see https://github.com/nschloe/meshio/blob/master/CHANGELOG.md
# mesh.cells_dict
# mesh.cell_data_dict

# ADDITIONAL METHODS (see https://github.com/nschloe/meshio/blob/master/meshio/_mesh.py#L123)
# mesh.get_cells_type("triangle")
# mesh.get_cell_data()