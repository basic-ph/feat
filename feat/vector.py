import numpy as np
from scipy import sparse

def compute_E_array(mesh, element_type, *materials):
    """
    Compute the array "E_array" containing the constitutive matrices data (6 entries) 
    of each element in the mesh. Normally the constituive matrix is a 3-by-3 matrix 
    but in linear elasticity we exploit its simmetry and save only the essential data:
    diagonal and the upper triangular part. This is useful for the following
    vectorized procedure.
    
    Parameters
    ----------
    mesh : meshio.Mesh
        Mesh obj with physical groups indicating materials
    element_type : str
        indentify the type of elements that compose the mesh
        it can be "triangle" or (not supported yet) "triangle6"
    *materials: feat.Material
        all Material objects present in the mesh (unpacked) 
    
    Returns
    -------
    E_array : (num_elements, 6) numpy.ndarray
        array containing constitutive matrix data for each element in the mesh
    """
    num_elements = mesh.cells_dict[element_type].shape[0]
    materials_num = len(materials)
    E_array = np.zeros((num_elements, 6))
    E_material = np.zeros((materials_num, 6)) # pre-computed array for each material
    material_map = mesh.cell_data_dict["gmsh:physical"][element_type] - 1  # element-material map

    for m in materials:
        tag = mesh.field_data[m.name][0] - 1   # convert to zero offset from unit offset (gmsh)
        E_material[tag] = m.E_flat
    
    E_array = E_material[material_map]
    return E_array


def compute_K_entry(row, col, c, e, E_array, t):
    """
    Evaluate the entry of the local stiffness matrix (K), identified by the local 
    indices (row, col), for all elements in the mesh using numpy vectorization.
    
    Parameters
    ----------
    row : int
        row index of the local stiffness matrix entry
    col : int
        column index of the local stiffness matrix entry
    c : (nodes_num, 2) numpy.ndarray
        cartesian coordinates of all nodes in the mesh expressed like: (x, y)
    e : (num_elements, nodes_per_element) numpy.ndarray
        elements map (connectivity map), n-th row contains tags of nodes in n-th element
    E_array : (num_elements, 6) numpy.ndarray
        array containing essential constitutive matrix entries for each element in the mesh
    t : float
        the thickness (z direction) of the 2D domain
    
    Returns
    -------
    k_data: (num_elements,) numpy.ndarray
        values of the entry K(row,col) of the local stiffness matrix for all elements
    """
    # Jacobian (determinant of Jacobian matrix) for all elements
    J = ((c[e[:,1]][:,0] - c[e[:,0]][:,0]) * (c[e[:,2]][:,1] - c[e[:,0]][:,1]) -
        (c[e[:,2]][:,0] - c[e[:,0]][:,0]) * (c[e[:,1]][:,1] - c[e[:,0]][:,1]))
    # compact form of the classic B matrix evaluated for all elements
    b = np.array([
        (c[e[:,1]][:,1] - c[e[:,2]][:,1], c[e[:,2]][:,0] - c[e[:,1]][:,0]),
        (c[e[:,2]][:,0] - c[e[:,1]][:,0], c[e[:,1]][:,1] - c[e[:,2]][:,1]),
        (c[e[:,2]][:,1] - c[e[:,0]][:,1], c[e[:,0]][:,0] - c[e[:,2]][:,0]),
        (c[e[:,0]][:,0] - c[e[:,2]][:,0], c[e[:,2]][:,1] - c[e[:,0]][:,1]),
        (c[e[:,0]][:,1] - c[e[:,1]][:,1], c[e[:,1]][:,0] - c[e[:,0]][:,0]),
        (c[e[:,1]][:,0] - c[e[:,0]][:,0], c[e[:,0]][:,1] - c[e[:,1]][:,1]),
    ])
    # data used for indexing E_array selecting the right entry of the constitutive matrix depending on row,col indices
    E_indices = np.array([(0, 1, 0, 1, 0, 1), (1, 3, 1, 3, 1, 3)])

    if (row % 2 == 0):
        E = E_array[:, E_indices[0, col]]
    else:
        E = E_array[:, E_indices[1, col]]
    # calculation of the values of that particular entry (row,col) for all elements
    k_data = (b[row][0] * b[col][0] * E + b[row][1] * b[col][1] * E_array[:,5]) / J * t * 0.5  # reminder: J/(J**2) = 1/J
    return k_data


def compute_global_dof(mesh, element_type, row, col):
    """
    Given the two LOCAL indices row and col, the function return two arrays containing 
    row GLOBAL indices and col GLOBAL indices for all elements. In other words it map 
    the local dof indices to global dof indices.
    
    Parameters
    ----------
    mesh : meshio.Mesh
        Mesh object
    element_type : str
        indentify the type of elements that compose the mesh
        it can be "triangle" or (not supported yet) "triangle6"
    row : int
        row index of the local stiffness matrix entry
    col : int
        column index of the local stiffness matrix entry
    
    Returns
    -------
    row_ind: (num_elements,) numpy.ndarray
        array of global row indices related to a certain local entry
    col_ind: (num_elements,) numpy.ndarray
        array of global column indices related to a certain local entry
    """
    elements = mesh.cells_dict[element_type]
    num_elements = mesh.cells_dict[element_type].shape[0]
    row_ind = np.zeros((num_elements))
    col_ind = np.zeros((num_elements))

    if (row % 2 == 0):
        row_ind = elements[:, row // 2] * 2
    else:
        row_ind = elements[:, row // 2] * 2 + 1
    
    if (col % 2 == 0):
        col_ind = elements[:, col // 2] * 2
    else:
        col_ind = elements[:, col // 2] * 2 + 1

    return row_ind, col_ind
    

def assembly(mesh, element_type, E_array, thickness):
    """
    Assemble the global sparse stiffness matrix of the system exploiting its simmetry.
    
    Parameters
    ----------
    mesh : meshio.Mesh
        Mesh object
    element_type : str
        indentify the type of elements that compose the mesh
        it can be "triangle" or (not supported yet) "triangle6"
    E_array : (num_elements, 6) numpy.ndarray
        array containing constitutive matrix data for each element in the mesh
    thickness : float
        the thickness (z direction) of the 2D domain
    
    Returns
    -------
    K : scipy.sparse.csc_matrix
        global stiffness matrix in Compressed Sparse Column format
    """

    t = thickness
    num_nodes = mesh.points.shape[0]
    num_elements = mesh.cells_dict[element_type].shape[0]
    # elements mapping, n-th row: nodes in n-th element
    elements = mesh.cells_dict["triangle"]
    coord = mesh.points[:,:2]  # x, y coordinates

    k_data = np.zeros((num_elements))
    row_ind = np.zeros((num_elements))
    col_ind = np.zeros((num_elements))
    K = sparse.csc_matrix((2 * num_nodes, 2 * num_nodes))
    
    # compute entries in the upper triangular matrix (without diagonal)
    for (row, col) in zip(*np.triu_indices(6, k=1)):  
        k_data = compute_K_entry(row, col, coord, elements, E_array, t)
        row_ind, col_ind = compute_global_dof(mesh, element_type, row, col)
        K += sparse.csc_matrix((k_data, (row_ind, col_ind)),shape=(2*num_nodes, 2*num_nodes))
    
    # copy previously computed entries in the lower triangular part
    K = K + K.transpose()

    # compute the diagonal entries
    for (row, col) in zip(*np.diag_indices(6)):
        k_data = compute_K_entry(row, col, coord, elements, E_array, t)
        row_ind, col_ind = compute_global_dof(mesh, element_type, row, col)
        K += sparse.csc_matrix((k_data, (row_ind, col_ind)),shape=(2*num_nodes, 2*num_nodes))

    return K
