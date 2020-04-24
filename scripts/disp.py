import meshio
import numpy as np

# disp = np.array([
#     [0.0, 0.0, 0.0],
#     [0.0, 1.0, 0.0],
#     [1.0, 1.0, 0.0],
#     [1.0, 0.0, 0.0],
#     [0.5, 0.5, 0.0],
# ])

mesh = meshio.read("./data/msh/base.msh")
print(mesh)
print(mesh.points)

D = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.5, 0.5])

# x_disp = d[::2]
# y_disp = a[1::2]

disp = np.column_stack((D[::2], D[1::2]))

disp_dict = {"disp": disp}

mesh.point_data = disp_dict

mesh.write("base_1.vtk")

