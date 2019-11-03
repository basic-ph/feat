# -*- coding: utf-8 -*-

import json
from pprint import pprint

import meshio
import numpy as np

from fem import stiffness_matrix, assembly, save_dirichlet_row, dirichlet_dof
from tools import compute_E_matrices, gauss_quadrature

from bc import DirichletBC, NeumannBC


# DATA
with open(r'../data/test_k_2.json', "r") as data_file:
        data = json.load(data_file)

element_type = data["element type"]
ip_number = data["integration points"]
thickness = data["thickness"]

# NUMERICAL INTEGRATION
weights, locations = gauss_quadrature(data)

# MESH
mesh = meshio.read(r"../gmsh/msh/test_k.msh")
nodal_coordinates = mesh.points[:,:2]  # slice is used to remove 3rd coordinate
nodes = mesh.points.shape[0]
dof = nodes * 2
connectivity_table = mesh.cells["triangle"]
elements = connectivity_table.shape[0]

# this array contains material tag for every element in mesh
element_material_map = mesh.cell_data["triangle"]["gmsh:physical"]
# print(element_material_map)
# print()

E_matrices = compute_E_matrices(data, mesh)
# pprint(E_matrices)
# print()

# arrays init
K = np.zeros((dof, dof))
R = np.zeros(dof)

for e in range(elements):
        k = stiffness_matrix(
                e,
                data,
                mesh,
                nodal_coordinates,
                connectivity_table,
                element_material_map,
                E_matrices
        )
        K = assembly(e, connectivity_table, k, K)

# K_saved = np.copy(K)  # FIXME now this is useless

print("K:\n", K)
print("R:\n", R)


left_side = DirichletBC("left side", data, mesh)
right_side = NeumannBC("right side", data, mesh)
# contrained dof rows of K are saved now
dirichlet_dof = dirichlet_dof(left_side)
print(dirichlet_dof)
K_rows = K[dirichlet_dof, :]
print(K_rows)


left_side.impose(K, R)
right_side.impose(R)
print("K:\n", K)
print("R:\n", R)

# Solution of the system
D = np.linalg.solve(K, R)
print("D:\n", D)

# reactions = np.dot(K_rows, D)
# print(reactions)
