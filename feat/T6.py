import numpy as np


x = lambda a, i, j: a[i][0] - a[j][0]
y = lambda b, i, j: b[i][1] - b[j][1]


def compute_B(c, w, z):
    # c (x,y) coordinate of 6 nodes
    #   [x1, y1],
    #   [x2, y2],
    #   [x3, y3],
    # weights 3 values for gauss integration
    #   [1/3, 1/3, 1/3]
    # locations [z1, z2, z3] triangular coordinate of IP
    #   [2/3, 1/6, 1/6]
    #   [1/6, 2/3, 1/6]
    #   [1/6, 1/6, 2/3]
    # see: http://kis.tu.kielce.pl//mo/COLORADO_FEM/colorado/IFEM.Ch24.pdf

    j = 0.5 * (x(c,1,0) * y(c,2,0) - y(c,0,1)*x(c,0,2))

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
    return B
