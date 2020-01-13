import sys

import numpy as np
from scipy import sparse
from numpy.linalg import det, inv


class Material():

    def __init__(self, gmsh_tag, young_modulus, poisson_ratio, load_condition):
        self.tag = gmsh_tag - 1  # convert to zero offset from unit offset (gmsh)
        self.young = young_modulus
        self.poisson = poisson_ratio

        if load_condition == "plane strain":
            # E_full: constitutive matrix for base implementation (3x3 2D array)
            coeff = self.young / ((1 + self.poisson) * (1 - 2 * self.poisson))
            matrix = np.array([
                [1 - self.poisson, self.poisson, 0],
                [self.poisson, 1 - self.poisson, 0],
                [0, 0, (1 - 2 * self.poisson) / 2]
            ])
            self.E_full = coeff * matrix  # 2D array: constitutive matrix (3x3)

            # E_flat: constitutive matrix for vectorized implementation
            # the original 3-by-3 matrix is reduced to saving 6 elements (diag+triu) in 1D array
            self.E_flat = np.array([
                self.young * (1 - self.poisson) / ((1 + self.poisson) * (1 - 2*self.poisson)),
                self.young * self.poisson / ((1 + self.poisson) * (1 - 2*self.poisson)),
                0.0,
                self.young * (1 - self.poisson) / ((1 + self.poisson) * (1 - 2*self.poisson)),
                0.0,
               self.young * (1 - self.poisson) / (2 * (1 + self.poisson) * (1 - 2*self.poisson)),
            ])

        elif load_condition == "plane stress":
            coeff = self.young / (1 - self.poisson ** 2)
            matrix = np.array([
                [1, self.poisson, 0],
                [self.poisson, 1, 0],
                [0, 0, (1 - self.poisson) / 2]
            ])
            self.E_full = coeff * matrix

            self.E_flat = np.array([
                self.young / (1 - self.poisson**2),
                self.young * self.poisson / (1 - self.poisson**2),
                0.0,
                self.young / (1 - self.poisson**2),
                0.0,
                self.young * (1 - self.poisson) / (2 * (1 - self.poisson**2)),
            ])


def compute_E_array(mesh, *materials):
    elements_num = mesh.cells["triangle"].shape[0]
    materials_num = len(materials)
    E_array = np.zeros((elements_num,3,3))  # 3D array composed by E matrix for each element
    E_material = np.zeros((materials_num,3,3))
    material_map = mesh.cell_data["triangle"]["gmsh:physical"] - 1  # element-material map

    for m in materials:
        # array containing constitutive matrices for each material in mesh
        E_material[m.tag,:,:] = m.E_full

    E_array = E_material[material_map]
    return E_array


def gauss_quadrature(element_type, integration_points):
    if element_type == "T3":
        if integration_points == 1:
            weights = np.array([1])
            locations = np.array([(1/3, 1/3)])
        elif integration_points == 3:  # only bulk rule
            weights = np.array([1/3, 1/3, 1/3])
            locations = np.array(
                [
                    (1/6, 1/6),
                    (2/3, 1/6),
                    (1/6, 2/3),
                ]
            )
    elif element_type == "T6":
        print("ERROR -- T6 element not implemented!!!")
        sys.exit()
    return weights, locations


x = lambda a, i, j: a[i][0] - a[j][0]
y = lambda b, i, j: b[i][1] - b[j][1]


def stiffness_matrix(e, mesh, E_array, thickness, element_type, integration_points):

    t = thickness
    element = mesh.cells["triangle"][e]
    # print("nodes:\n", element)
    c = mesh.points[:,:2][element]
    # print("coord:\n", c)
 
    E = E_array[e]
    # print("E:\n", E)

    # element/local stiffness matrix
    k = np.zeros((6, 6))  # for T6 --> 12 by 12

    weights, locations = gauss_quadrature(element_type, integration_points)
    # print(weights)
    # print(locations)
    for p in range(weights.shape[0]):
        w = weights[p]
        loc = locations[p]  # this is a [x, y] array

        j = ( (c[1][0] - c[0][0]) * (c[2][1] - c[0][1])  # det of jacobian matrix
            - (c[2][0] - c[0][0]) * (c[1][1] - c[0][1])
        )
        # print("j",j)
        B = (1/j) * np.array([
            (y(c, 1, 2), 0, y(c, 2, 0), 0, y(c, 0, 1), 0),
            (0, x(c, 2, 1), 0, x(c, 0, 2), 0 , x(c, 1, 0)),
            (x(c, 2, 1), y(c, 1, 2), x(c, 0, 2), y(c, 2, 0), x(c, 1, 0), y(c, 0, 1))
        ])
        # print("B", B)
        k_integral = B.T @ E @ B * t * 0.5 * j * w
        k += k_integral

    return k


def compute_global_dof(e, mesh):
    element = mesh.cells["triangle"][e]
    element_dof = np.zeros(6, dtype=np.int32)  # becomes 12 for T6
    for n in range(element.shape[0]):  # TODO check if applicable for BC
        element_dof[n*2] = element[n] * 2
        element_dof[n*2+1] = element[n] * 2 + 1
    return element_dof


def assembly(K, elements_num, mesh, E_array, thickness, element_type, integration_points):

    for e in range(elements_num):
        k = stiffness_matrix(e, mesh, E_array, thickness, element_type, integration_points)
        element_dof = compute_global_dof(e, mesh)
        for i in range(6):  # becomes 12 for T6
            I = element_dof[i]
            for j in range(6):  # becomes 12 for T6
                J = element_dof[j]
                K[I, J] += k[i, j]
    return K


def sparse_assembly(K, elements_num, mesh, E_array, thickness, element_type, integration_points):
    nodes = mesh.points.shape[0]
    for e in range(elements_num):
        k = stiffness_matrix(e, mesh, E_array, thickness, element_type, integration_points)
        k_data = np.ravel(k)  # flattened 6x6 local matrix
        element_dof = compute_global_dof(e, mesh)

        row_ind = np.repeat(element_dof, 6)
        col_ind = np.tile(element_dof, 6)
        K += sparse.csc_matrix((k_data, (row_ind, col_ind)),shape=(2*nodes, 2*nodes))
    return K


def compute_modulus(mesh, boundary, reactions, t):

    # pick x coord of first point in boundary
    lenght = mesh.points[boundary.nodes[0]][0]
    x_disp = boundary.value  # x displacement of node 1
    strain = x_disp / lenght
    avg_stress = abs(np.sum(reactions) / (lenght * t))
    modulus = avg_stress / strain

    return modulus