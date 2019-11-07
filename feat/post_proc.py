import numpy as np


def compute_modulus(mesh, boundary, reactions, t):

    # pick x coord of first point in boundary
    lenght = mesh.points[boundary.nodes[0]][0]
    x_disp = boundary.imposed_disp  # x displacement of node 1
    strain = x_disp / lenght
    avg_stress = abs(np.sum(reactions) / (lenght * t))
    modulus = avg_stress / strain

    return modulus
