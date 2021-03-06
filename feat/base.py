import sys
import logging
import numpy as np
from scipy import sparse
from numpy.linalg import det, inv


logger = logging.getLogger(__name__)

class Material():

    def __init__(self, name, young_modulus, poisson_ratio, load_condition):
        self.name = name
        self.young = young_modulus
        self.poisson = poisson_ratio
        logger.debug("Material %s loaded: E = %s, nu = %s", self.name, self.young, self.poisson)

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
            # the original 3-by-3 matrix is reduced to saving 6 elem (diag+triu) in 1D array
            self.E_flat = np.array([
                self.young * (1 - self.poisson) / ((1 + self.poisson) * (1 - 2*self.poisson)),
                self.young * self.poisson / ((1 + self.poisson) * (1 - 2*self.poisson)),
                0.0,
                self.young * (1 - self.poisson) / ((1 + self.poisson) * (1 - 2*self.poisson)),
                0.0,
               self.young * (1 -  2*self.poisson) / (2 * (1 + self.poisson) * (1 - 2*self.poisson)),
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


def compute_E_material(num_elements, material_map, field_data, *materials):
    num_materials = len(materials)
    E_mat = np.zeros((num_materials,3,3)) # 3D array composed by E matrix for each material

    for m in materials:
        tag = field_data[m.name][0] - 1   # convert to zero offset from unit offset (gmsh)
        # array containing constitutive matrices for each material in mesh
        E_mat[tag,:,:] = m.E_full

    return E_mat


x = lambda a, i, j: a[i][0] - a[j][0]
y = lambda b, i, j: b[i][1] - b[j][1]


def stiffness_matrix(e, elem, coord, mat_map, E_mat, h, elem_type):
    t = h
    element = elem[e]
    c = coord[element]  # indexing with an array

    material_tag = mat_map[e]
    E = E_mat[material_tag]

    # element/local stiffness matrix
    k = np.zeros((2*element.shape[0], 2*element.shape[0]))  # for T6 --> 12 by 12

    j = ( (c[1][0] - c[0][0]) * (c[2][1] - c[0][1])  # det of jacobian matrix
        - (c[2][0] - c[0][0]) * (c[1][1] - c[0][1])
    )
    # j = (x(c,1,0) * y(c,2,0) - y(c,0,1)*x(c,0,2))
    B = (1/j) * np.array([
        (y(c, 1, 2), 0, y(c, 2, 0), 0, y(c, 0, 1), 0),
        (0, x(c, 2, 1), 0, x(c, 0, 2), 0 , x(c, 1, 0)),
        (x(c, 2, 1), y(c, 1, 2), x(c, 0, 2), y(c, 2, 0), x(c, 1, 0), y(c, 0, 1))
    ])
    k = B.T @ E @ B * t * 0.5 * j

    return k


def compute_global_dof(e, elem, elem_type):
    element = elem[e]
    element_dof = np.zeros(6, dtype=np.int32)  # becomes 12 for T6
    for n in range(element.shape[0]):  # TODO check if applicable for BC
        element_dof[n*2] = element[n] * 2
        element_dof[n*2+1] = element[n] * 2 + 1
    return element_dof


def assembly(K, num_elem, elem, coord, mat_map, E_mat, h, elem_type):

    for e in range(num_elem):
        k = stiffness_matrix(e, elem, coord, mat_map, E_mat, h, elem_type)
        element_dof = compute_global_dof(e, elem, elem_type)
        for i in range(2 * elem[0].shape[0]):  # range(6)
            I = element_dof[i]
            for j in range(2 * elem[0].shape[0]):  # range(6)
                J = element_dof[j]
                K[I, J] += k[i, j]
    return K


def sp_assembly(K, num_elem, num_nodes, elem, coord, mat_map, E_mat, h, elem_type):
    data_tmp = []
    row_data = []
    col_data = []
    for e in range(num_elem):
        k = stiffness_matrix(e, elem, coord, mat_map, E_mat, h, elem_type)
        k_data = np.ravel(k)  # flattened 6x6 local matrix
        data_tmp.append(k_data)

        element_dof = compute_global_dof(e, elem, elem_type)
        row_ind = np.repeat(element_dof, 6)
        col_ind = np.tile(element_dof, 6)
        row_data.append(row_ind)
        col_data.append(col_ind)

    row = np.concatenate(row_data)
    col = np.concatenate(col_data)
    data = np.concatenate(data_tmp)
    K = sparse.coo_matrix((data,(row, col)), shape=(2*num_nodes, 2*num_nodes))
    K = K.tocsc()
    return K


def compute_modulus(coord, boundary, reactions, t):

    # pick x coord of first point in boundary
    lenght = coord[boundary.nodes[0]][0]
    x_disp = boundary.value  # x displacement of node 1
    strain = x_disp / lenght
    avg_stress = abs(np.sum(reactions) / (lenght * t))
    modulus = avg_stress / strain
    return modulus