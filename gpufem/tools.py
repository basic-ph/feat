import sys

import numpy as np
from pprint import pprint


def compute_E_matrices(data, mesh):
    condition = data["load condition"]
    E_matrices = {}

    for key,value in data["materials"].items():
        # key is the material name
        # value is the dict with young's modulus and poisson's ratio
        physical_tag = mesh.field_data[key][0]
        
        poisson = value["poisson's ratio"]
        young = value["young's modulus"]

        if condition == "plane strain":
            coeff = young / ((1 + poisson) * (1 - 2 * poisson))
            matrix = np.array([
                [1 - poisson, poisson, 0],
                [poisson, 1 - poisson, 0],
                [0, 0, (1 - 2 * poisson) / 2]
            ])
            E = coeff * matrix
        elif condition == "plane stress":
            coeff = young / (1 - poisson ** 2)
            matrix = np.array([
                [1, poisson, 0],
                [poisson, 1, 0],
                [0, 0, (1-poisson)/2]
            ])
            E = coeff * matrix

        E_matrices[physical_tag] = {}
        E_matrices[physical_tag]["name"] = key
        E_matrices[physical_tag]["E"] = E

    return E_matrices


def gauss_quadrature(data):
    if data["element type"] == "T3":
        if data["integration points"] == 1:
            weights = np.array([1])
            locations = np.array([(1/3, 1/3)])
        elif data["integration points"] == 3:  # only bulk rule
            weights = np.array([1/3, 1/3, 1/3])
            locations = np.array(
                [
                    (1/6, 1/6),
                    (2/3, 1/6),
                    (1/6, 2/3),
                ]
            )
    elif data["element type"] == "T6":
        print("ERROR -- T6 element not implemented!!!")
        sys.exit()
    return weights, locations
