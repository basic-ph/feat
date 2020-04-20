
def stiffness_matrix(e, elements, nodal_coord, E_material, thickness, element_type, integration_points):
    t = thickness
    element = elements[e]
    c = nodal_coord[element]  # indexing with an array

    material_tag = material_map[e]
    E = E_material[material_tag]
    
def sp_assembly(K, num_elements, num_nodes, elements, nodal_coord, material_map, E_material, thickness, element_type, integration_points):


elements = mesh.cells_dict[element_type]
nodal_coord = mesh.points[:,:2]
num_elements = elements.shape[0]
num_nodes = nodal_coord.shape[0]
material_map = mesh.cell_data_dict["gmsh:physical"][element_type] - 1  # element-material map

def compute_global_dof(num_elements, elements, row, col):