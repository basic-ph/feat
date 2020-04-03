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
            # the original 3-by-3 matrix is reduced to saving 6 elements (diag+triu) in 1D array
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


def compute_E_material(mesh, element_type, *materials):
    """
    Compute the array "E_array" containing the constitutive matrices (3x3) of each 
    element in the mesh.
    
    Parameters
    ----------
    mesh : meshio.Mesh
        Mesh object with physical groups indicating materials
    element_type : str
        indentify the type of elements that compose the mesh
        it can be "triangle" or (not supported yet) "triangle6"
    
    Returns
    -------
    E_array : (elements_num, 3, 3) numpy.ndarray
        three-dimensional array with as many "pages" as elements in the mesh
    """
    elements_num = mesh.cells_dict[element_type].shape[0]
    materials_num = len(materials)
    E_material = np.zeros((materials_num,3,3)) # 3D array composed by E matrix for each material
    material_map = mesh.cell_data_dict["gmsh:physical"][element_type] - 1  # element-material map

    for m in materials:
        tag = mesh.field_data[m.name][0] - 1   # convert to zero offset from unit offset (gmsh)
        # array containing constitutive matrices for each material in mesh
        E_material[tag,:,:] = m.E_full

    return E_material


def gauss_quadrature(element_type, integration_points):
    if element_type == "triangle":
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
    elif element_type == "triangle6":
        if integration_points == 3:
            weights = np.array([1/3, 1/3, 1/3])
            locations = np.array([
                [2/3, 1/6, 1/6],
                [1/6, 2/3, 1/6],
                [1/6, 1/6, 2/3],
            ])
            # locations = np.array([
            #     [1/2, 1/2, 0.0],
            #     [0.0, 1/2, 1/2],
            #     [1/2, 0.0, 1/2],
            # ])
        else:
            print("ERROR -- for T6 only 3 point rule is supported!!!")
            sys.exit()
    return weights, locations


x = lambda a, i, j: a[i][0] - a[j][0]
y = lambda b, i, j: b[i][1] - b[j][1]


def stiffness_matrix(e, mesh, E_material, thickness, element_type, integration_points):

    t = thickness
    element = mesh.cells_dict[element_type][e]
    # print("nodes:\n", element.shape[0])
    c = mesh.points[:,:2][element]
    # print("coord:\n", c)
 
    e_tag = mesh.cell_data_dict["gmsh:physical"][element_type][e] - 1  # physical tag relative to element e (identify material)
    E = E_material[e_tag]

    # element/local stiffness matrix
    k = np.zeros((2*element.shape[0], 2*element.shape[0]))  # for T6 --> 12 by 12
    # print("k shape ", k.shape)

    weights, locations = gauss_quadrature(element_type, integration_points)
    # print(weights)
    # print(locations)
    if element_type == "triangle":
        w = weights[0]
        j = ( (c[1][0] - c[0][0]) * (c[2][1] - c[0][1])  # det of jacobian matrix
            - (c[2][0] - c[0][0]) * (c[1][1] - c[0][1])
        )
        B = (1/j) * np.array([
            (y(c, 1, 2), 0, y(c, 2, 0), 0, y(c, 0, 1), 0),
            (0, x(c, 2, 1), 0, x(c, 0, 2), 0 , x(c, 1, 0)),
            (x(c, 2, 1), y(c, 1, 2), x(c, 0, 2), y(c, 2, 0), x(c, 1, 0), y(c, 0, 1))
        ])
        k = B.T @ E @ B * t * 0.5 * j

    elif element_type == "triangle6":
        np.set_printoptions(linewidth=200)
        print("calculating k local for triangle 6")
        for p in range(integration_points):  # weights.shape[0]
            w = weights[p]
            z = locations[p]  # location in triangular coord: [z1, z2, z3]
            # print("IP number", p)
            # print("weight", w)
            # print("loc", z)
            # print("loc", z[0], z[1], z[2])

            j = 0.5 * (x(c,1,0) * y(c,2,0) - y(c,0,1)*x(c,0,2))
            # print("j", j)

            dNx1 = (4*z[0] - 1) * y(c,1,2) / (2 * j)
            dNx2 = (4*z[1] - 1) * y(c,2,0) / (2 * j)
            dNx3 = (4*z[2] - 1) * y(c,0,1) / (2 * j)
            dNx4 = 4 * (z[1] * y(c,1,2) + z[0] * y(c,2,0)) / (2 * j)
            dNx5 = 4 * (z[2] * y(c,2,0) + z[1] * y(c,0,1)) / (2 * j)
            dNx6 = 4 * (z[0] * y(c,0,1) + z[2] * y(c,1,2)) / (2 * j)

            dNy1 = (4*z[0] - 1) * x(c,2,1) / (2 * j)
            dNy2 = (4*z[0] - 1) * x(c,0,2) / (2 * j)
            dNy3 = (4*z[0] - 1) * x(c,1,0) / (2 * j)
            dNy4 = 4 * (z[1] * x(c,2,1) + z[0] * x(c,0,2)) / (2 * j)
            dNy5 = 4 * (z[2] * x(c,0,2) + z[1] * x(c,1,0)) / (2 * j)
            dNy6 = 4 * (z[0] * x(c,1,0) + z[2] * x(c,2,1)) / (2 * j)

            B = np.array([
                [dNx1, 0, dNx2, 0, dNx3, 0, dNx4, 0, dNx5, 0, dNx6, 0],
                [ 0, dNy1, 0, dNy2, 0, dNy3, 0, dNy4, 0, dNy5, 0, dNy6],
                [dNy1, dNx1, dNy2, dNx2, dNy3, dNx3, dNy4, dNx4, dNy5, dNx5, dNy6, dNx6]
            ])

            k_integral = B.T @ E @ B * t * j * w
            # print("k integral\n", k_integral)
            k += k_integral

    return k


def compute_global_dof(e, mesh, element_type):
    element = mesh.cells_dict[element_type][e]
    element_dof = np.zeros(6, dtype=np.int32)  # becomes 12 for T6
    for n in range(element.shape[0]):  # TODO check if applicable for BC
        element_dof[n*2] = element[n] * 2
        element_dof[n*2+1] = element[n] * 2 + 1
    return element_dof


def assembly(K, elements_num, mesh, E_material, thickness, element_type, integration_points):

    for e in range(elements_num):
        k = stiffness_matrix(e, mesh, E_material, thickness, element_type, integration_points)
        element_dof = compute_global_dof(e, mesh, element_type)
        for i in range(6):  # becomes 12 for T6
            I = element_dof[i]
            for j in range(6):  # becomes 12 for T6
                J = element_dof[j]
                K[I, J] += k[i, j]
    return K


def sparse_assembly(K, elements_num, mesh, E_material, thickness, element_type, integration_points):
    nodes = mesh.points.shape[0]
    for e in range(elements_num):
        k = stiffness_matrix(e, mesh, E_material, thickness, element_type, integration_points)
        k_data = np.ravel(k)  # flattened 6x6 local matrix
        element_dof = compute_global_dof(e, mesh, element_type)

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