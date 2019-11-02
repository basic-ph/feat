# -*- coding: utf-8 -*-
"""
meshio.read >> read_buffer (the one in msh4_1 module)

    return Mesh(
        points, -- "Nodes" 
        cells, -- "Elements"
        point_data=point_data, -- $NodeData section (not used)
        cell_data=cell_data, -- "PhysicalTag"
        field_data=field_data, -- "PhysicalNames"
        gmsh_periodic=periodic, (not used)
    )
"""
import meshio


mesh = meshio.read(r"gmsh\msh\bar.msh")
print("mesh.points:\n", mesh.points)
print()
print("mesh.cells:\n", mesh.cells)
print()
print("mesh.cell_data:", mesh.cell_data)
print()
print("mesh.field_data: ", mesh.field_data)


# ------------------------------------------------------------------------------ 
# mesh.points: array dei nodi con loro coordinate
#  [[0. 0. 0.]
#  [2. 0. 0.]
#  [1. 0. 0.]]

# mesh.cells:dict con elementi divisi per tipo e array con nodi (zero-based) per ciascun elemento
#  {'vertex': array([[0],
#        [1]]), 'line': array([[0, 2],
#        [2, 1]])}

# mesh.cell_data: PhysicalTag degli elementi suddivisi per tipo
# {'vertex': {'gmsh:physical': array([1, 2])}, 'line': {'gmsh:physical': array([3, 3])}}

# mesh.field_data: PhysicalNames mappa il nome, la PhysicalTag e la dimensione
# {'D': array([1, 0]), 'N': array([2, 0]), 'bar': array([3, 1])}
# ------------------------------------------------------------------------------