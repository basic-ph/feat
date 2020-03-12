import pygmsh
import meshio

geom = pygmsh.built_in.Geometry()

side = 2.0
lcar = 2.0

p1 = geom.add_point([0.0, 0.0, 0.0], lcar)
p2 = geom.add_point([side, 0.0, 0.0], lcar)
p3 = geom.add_point([side, side, 0.0], lcar)
p4 = geom.add_point([0.0, side, 0.0], lcar)

l1 = geom.add_line(p1, p2)
l2 = geom.add_line(p2, p3)
l3 = geom.add_line(p3, p4)
l4 = geom.add_line(p4, p1)

loop1 = geom.add_line_loop([l1, l2, l3, l4])
square = geom.add_plane_surface(loop1)

geom.add_physical(square, label="cheese")
geom.add_physical(l4, label="left side")
geom.add_physical(p1, label="bottom left corner")
geom.add_physical(l2, label="right side")

mesh = pygmsh.generate_mesh(geom)

print("mesh.points:\n", mesh.points)
print()
print("mesh.cells_dict:\n", mesh.cells_dict)
print()
print("mesh.cell_data_dict:", mesh.cell_data_dict)
print()
print("mesh.field_data: ", mesh.field_data)