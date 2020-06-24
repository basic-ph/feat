import numpy as np
from scipy import sparse
import logging

logger = logging.getLogger(__name__)


def compute_E_array(num_elem, mat_map, field_data, *materials):
    """
    Compute the array "E_array" containing the constitutive matrices data (6 entries) 
    of each element in the mesh. Normally the constituive matrix is a 3-by-3 matrix 
    but in linear elasticity we exploit its simmetry and save only the essential data:
    diagonal and the upper triangular part. This is useful for the following
    vectorized procedure.
    
    Parameters
    ----------
    num_elem : int
        number of elem in mesh
    mat_map : numpy.ndarray
        array containing material tag (zero offset) for evey element in mesh
    field_data: dict
        field_data attribute of meshio.Mesh object containing physical group tags
    *materials: feat.Material
        all Material objects present in the mesh (unpacked) 
    
    Returns
    -------
    E_array : (num_elem, 6) numpy.ndarray
        array containing constitutive matrix data for each element in the mesh
    """
    num_materials = len(materials)
    E_array = np.zeros((num_elem, 6))
    E_material = np.zeros((num_materials, 6)) # pre-computed array for each material

    for m in materials:
        tag = field_data[m.name][0] - 1   # convert to zero offset from unit offset (gmsh)
        E_material[tag] = m.E_flat
    
    E_array = E_material[mat_map]
    return E_array


X = lambda c, e, i, j: c[e[:,i]][:,0] - c[e[:,j]][:,0]
Y = lambda c, e, i, j: c[e[:,i]][:,1] - c[e[:,j]][:,1]


def compute_K_entry(row, col, c, e, b, J, E_array, t):

    A = row % 2
    B = row + (-row // 2)
    C = col % 2
    D = col + (-col // 2)

    E = int(row % 2 == 0)
    F = (row + (-1)**row) + (-(row + (-1)**row)//2)
    G = int(col % 2 == 0)
    H = (col + (-1)**col) + (-(col + (-1)**col)//2)

    k_data = (b[A,B] * b[C,D] * E_array[:,(row+col) % 2] + b[E,F] * b[G,H] * E_array[:,5]) / J * t * 0.5
    return k_data


def compute_global_dof(num_elem, elem, row, col):
    """
    Given the two LOCAL indices row and col, the function return two arrays containing 
    row GLOBAL indices and col GLOBAL indices for all elem. In other words it map 
    the local dof indices to global dof indices.
    
    Parameters
    ----------
    num_elem : int
        number of elem in mesh
    elem : (num_elem, nodes_per_element) numpy.ndarray
        elem map (connectivity map), n-th row contains tags of nodes in n-th element 
    row : int
        row index of the local stiffness matrix entry
    col : int
        column index of the local stiffness matrix entry
    
    Returns
    -------
    row_ind: (num_elem,) numpy.ndarray
        array of global row indices related to a certain local entry
    col_ind: (num_elem,) numpy.ndarray
        array of global column indices related to a certain local entry
    """

    if (row % 2 == 0):
        row_ind = elem[:, row // 2] * 2
    else:
        row_ind = elem[:, row // 2] * 2 + 1
    
    if (col % 2 == 0):
        col_ind = elem[:, col // 2] * 2
    else:
        col_ind = elem[:, col // 2] * 2 + 1

    return row_ind, col_ind
    

def assembly(num_elem, num_nod, elem, coord, E_array, h):
    """
    Assemble the global sparse stiffness matrix of the system exploiting its simmetry.
    
    Parameters
    ----------
    num_elem : int
        number of elem in mesh
    num_nod : int
        number of nodes in mesh
    elem : (num_elem, nodes_per_element) numpy.ndarray
        elem map (connectivity map), n-th row contains tags of nodes in n-th element
    coord : (nodes_num, 2) numpy.ndarray
        cartesian coordinates of all nodes in the mesh expressed like: (x, y)
    E_array : (num_elem, 6) numpy.ndarray
        array containing constitutive matrix data for each element in the mesh
    h : float
        the thickness (z direction) of the 2D domain
    
    Returns
    -------
    K : scipy.sparse.csc_matrix
        global stiffness matrix in Compressed Sparse Column format
    """
    c = coord
    e = elem
    J = X(c,e,1,0) * Y(c,e,2,0) - X(c,e,2,0) * Y(c,e,1,0)

    b = np.array([
        [Y(c,e,1,2), Y(c,e,2,0), Y(c,e,0,1)],
        [X(c,e,2,1), X(c,e,0,2), X(c,e,1,0)],
    ])
    # logger.debug("b shape: %s", b.shape)
    # logger.debug("b bytes: %s", b.nbytes)

    K = sparse.csc_matrix((2 * num_nod, 2 * num_nod))
    
    # compute entries in the upper triangular matrix (without diagonal)
    for (row, col) in zip(*np.triu_indices(6, k=1)):  
        k_data = compute_K_entry(row, col, coord, elem, b, J, E_array, h)
        row_ind, col_ind = compute_global_dof(num_elem, elem, row, col)
        K += sparse.csc_matrix((k_data, (row_ind, col_ind)),shape=(2*num_nod, 2*num_nod))
    
    # copy previously computed entries in the lower triangular part
    K = K + K.transpose()

    # compute the diagonal entries
    for (row, col) in zip(*np.diag_indices(6)):
        k_data = compute_K_entry(row, col, coord, elem, b, J, E_array, h)
        row_ind, col_ind = compute_global_dof(num_elem, elem, row, col)
        K += sparse.csc_matrix((k_data, (row_ind, col_ind)),shape=(2*num_nod, 2*num_nod))

    return K
