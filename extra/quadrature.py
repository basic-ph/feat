# -*- coding: utf-8 -*-

import numpy
import quadpy

# points, weights = np.polynomial.legendre.leggauss(4)
# print()
# print(points)
# print()
# print(weights)

scheme = quadpy.triangle.centroid()
print(scheme.points)