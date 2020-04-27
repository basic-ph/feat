import numpy as np
from numpy.lib.arraysetops import unique


class BoundaryCondition():
    
    def __init__(self, name, mesh, dof, value):
        self.name = name
        self.tag = mesh.field_data[name][0]
        self.dim = mesh.field_data[name][1]
        self.local_dof = np.asarray(dof)
        self.value = value
        if self.dim == 0:
            # array containing indices of elements in the boundary
            self.elements = np.nonzero(mesh.cell_data_dict["gmsh:physical"]["vertex"] == self.tag)[0]
            # array containing indices of nodes in the boundary
            self.nodes = unique(mesh.cells_dict["vertex"][self.elements])
        elif self.dim == 1:
            self.elements = np.nonzero(mesh.cell_data_dict["gmsh:physical"]["line"] == self.tag)[0]
            self.nodes = unique(mesh.cells_dict["line"][self.elements])

    def compute_global_dof(self, nodes, local_dof, dof_num):
        global_dof = np.zeros(dof_num, dtype=np.int32)
        i = 0
        for n in nodes:
            for d in local_dof:
                dof = n * 2 + d
                global_dof[i] = dof
                i += 1
        return global_dof


class DirichletBC(BoundaryCondition):

    def __init__(self, name, mesh, dof, value):
        super().__init__(name, mesh, dof, value)
        # read list of constrained dof in this bc
        self.global_dof_num = self.nodes.shape[0] * self.local_dof.shape[0]
        self.global_dof = super(DirichletBC, self).compute_global_dof(self.nodes, self.local_dof, self.global_dof_num)
        self.values = np.repeat(self.value, self.global_dof_num)


class NeumannBC(BoundaryCondition):

    def __init__(self, name, mesh, dof, value):
        super().__init__(name, mesh, dof, value)
        self.global_dof_num = self.nodes.shape[0] * self.local_dof.shape[0]
        self.global_dof = super(NeumannBC, self).compute_global_dof(self.nodes, self.local_dof, self.global_dof_num)
        # nodal load = total load / number of nodes in this boundary
        self.nodal_load = self.value / self.nodes.shape[0]
        

def dirichlet_dof(*conditions):  # FIXME
    array_list = [c.global_dof for c in conditions]
    total_dof = np.concatenate(array_list)
    # is necessary to add check for duplicate?
    return total_dof


def build_dirichlet_data(*conditions):
    dir_dof_list = [c.global_dof for c in conditions]
    dir_values_list = [c.values for c in conditions]

    dir_dof = np.concatenate(dir_dof_list)
    dir_values = np.concatenate(dir_values_list)

    return dir_dof, dir_values


def apply_dirichlet(K, R, *conditions): 
    dir_dof, dir_values = build_dirichlet_data(*conditions)

    # loop over all dirichlet dof
    for i in range(dir_dof.shape[0]):
        R -= dir_values[i] * K[:, dir_dof[i]]  # modify R (force vector)
    
    for j in range(dir_dof.shape[0]):
        R[dir_dof[j]] = dir_values[j]  # enforce dirichlet value related to that dof
        K[:, dir_dof[j]] = 0.0  # clearing column related to dirichlet dof
        K[dir_dof[j], :] = 0.0  # clearing row...
        K[dir_dof[j], dir_dof[j]] = 1.0  # transform in identity row

    return K, R


def apply_neumann(R, *conditions):
    for c in conditions:
        for d in c.global_dof:
            R[d] += c.nodal_load
    return R


def sp_apply_dirichlet(nodes, K, R, *conditions):
    # TODO check if K is in CSC format!!!
    dir_dof, dir_values = build_dirichlet_data(*conditions)
    R[dir_dof] = dir_values
    # mask of booleans checking if (row) indices are present in dirichlet dof array
    mask_csc = np.in1d(K.indices, dir_dof)
    K.data[mask_csc] = 0.0  # elements are cleared directly from data sparse attribute
    K[dir_dof, dir_dof] = 1.0  # fancy indexing

    return K, R