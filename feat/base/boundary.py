import numpy as np
from numpy.lib.arraysetops import unique


def dirichlet_dof(*conditions):
    array_list = [c.global_dof for c in conditions]
    total_dof = np.concatenate(array_list)
    # is necessary to add check for duplicate?
    return total_dof


class BoundaryCondition():
    
    def __init__(self, name, mesh, dof, value):
        self.name = name
        self.tag = mesh.field_data[name][0]
        self.dim = mesh.field_data[name][1]
        self.local_dof = np.asarray(dof)
        self.value = value
        if self.dim == 0:
            # array containing indices of elements in the boundary
            self.elements = np.nonzero(mesh.cell_data["vertex"]["gmsh:physical"] == self.tag)[0]
            # array containing indices of nodes in the boundary
            self.nodes = unique(mesh.cells["vertex"][self.elements])
        elif self.dim == 1:
            self.elements = np.nonzero(mesh.cell_data["line"]["gmsh:physical"] == self.tag)[0]
            self.nodes = unique(mesh.cells["line"][self.elements])

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

    def impose(self, K, R):
        for d in self.global_dof:
            R -= self.value * K[:, d]  # modify RHS
            K[:, d] = 0.0  # zero-out column
            K[d, :] = 0.0  # zero-out row
            K[d, d] = 1.0  # set diagonal to 1
            R[d] = self.value  # enforce value
            
    def sparse_impose(self, K, R):
        for d in self.global_dof:
            K_col = np.ravel(K[:,d].toarray())  # FIXME is this efficient??
            R -= self.value * K_col  # modify RHS
            K[:, d] = 0.0  # zero-out column
            K[d, :] = 0.0  # zero-out row
            K[d, d] = 1.0  # set diagonal to 1
            R[d] = self.value  # enforce value


class NeumannBC(BoundaryCondition):

    def __init__(self, name, mesh, dof, value):
        super().__init__(name, mesh, dof, value)
        self.global_dof_num = self.nodes.shape[0] * self.local_dof.shape[0]
        self.global_dof = super(NeumannBC, self).compute_global_dof(self.nodes, self.local_dof, self.global_dof_num)
        # print('name', self.name)
        # print('neu glob dof:', self.global_dof)
        
    def impose(self, R):
        for d in self.global_dof:
            # nodal load = total load / number of nodes in this boundary
            self.nodal_load = self.value / self.nodes.shape[0]
            R[d] += self.nodal_load