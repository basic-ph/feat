import numpy as np
from numpy.lib.arraysetops import unique

class BoundaryCondition():
    
    def __init__(self, name, data, mesh):
        self.name = name
        self.tag = mesh.field_data[name][0]
        # array containing indices of elements in the boundary
        self.elements = np.nonzero(mesh.cell_data["line"]["gmsh:physical"] == self.tag)[0]
        # array containing indices of nodes in the boundary
        self.nodes = unique(mesh.cells["line"][self.elements])
        
        print(self.name)
        print(self.nodes)

    def compute_global_dof(self, nodes, local_dof):
        dof_num = nodes.shape[0] * local_dof.shape[0]
        global_dof = np.zeros(dof_num, dtype=np.int32)
        i = 0
        for n in nodes:
            for d in local_dof:
                dof = n * 2 + d
                global_dof[i] = dof
                i += 1
        return global_dof


class DirichletBC(BoundaryCondition):

    def __init__(self, name, data, mesh):
        super().__init__(name, data, mesh)
        # read list of constrained dof in this bc
        self.constrained_dof = np.asarray(data["bc"]["dirichlet"][name]["dof"])
        self.imposed_disp = data["bc"]["dirichlet"][name]["value"]
        print(self.constrained_dof)
        print(self.imposed_disp)
        
        self.global_dof = super(DirichletBC, self).compute_global_dof(self.nodes, self.constrained_dof)
        print(self.global_dof)

    def impose(self, K, R):
        for d in self.global_dof:
            R -= self.imposed_disp * K[:, d]  # modify RHS
            K[:, d] = 0.0  # zero-out column
            K[d, :] = 0.0  # zero-out row
            K[d, d] = 1.0  # set diagonal to 1
            R[d] = self.imposed_disp  # enforce value




class NeumannBC(BoundaryCondition):

    def __init__(self, name, data, mesh):
        super().__init__(name, data, mesh)
        self.constrained_dof = np.asarray(data["bc"]["neumann"][name]["dof"])
        self.imposed_load = data["bc"]["neumann"][name]["value"]
        print(self.constrained_dof)
        print(self.imposed_load)
        
        self.global_dof = super(NeumannBC, self).compute_global_dof(self.nodes, self.constrained_dof)
        print(self.global_dof)

    def impose(self, R):
        for d in self.global_dof:
            # nodal load = total load / number of nodes in this boundary
            self.nodal_load = self.imposed_load / self.nodes.shape[0]
            R[d] += self.nodal_load