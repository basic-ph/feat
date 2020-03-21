import numpy as np


x = lambda a, i, j: a[i][0] - a[j][0]
y = lambda b, i, j: b[i][1] - b[j][1]


def stiffness_matrix(e, mesh, E_array, thickness, element_type, integration_points):

    t = thickness
    element = mesh.cells_dict[element_type][e]
    print("nodes:\n", element.shape[0])
    c = mesh.points[:,:2][element]
    print("coord:\n", c)
 
    E = E_array[e]
    print("E:\n", E)

    # element/local stiffness matrix
    k = np.zeros((2*element.shape[0], 2*element.shape[0]))  # for T6 --> 12 by 12
    print("k shape ", k.shape)

    weights = np.array([1/3, 1/3, 1/3])
    locations = np.array([
                [2/3, 1/6, 1/6],
                [1/6, 2/3, 1/6],
                [1/6, 1/6, 2/3],
            ])

    print("calculating k local for triangle 6")
    np.set_printoptions(linewidth=200)
    
    A = 0.5 * (x(c,1,0) * y(c,2,0) - y(c,0,1)*x(c,0,2))  # j

    B1 = (1 / 2*A) * np.array([
        [y(c,2,1),   0.0   , y(c,2,0),   0.0   , y(c,0,1),   0.0   , 2*y(c,1,2),   0.0   , 2*y(c,2,1),   0.0   , 2*y(c,1,2),   0.0   ],
        [   0.0  , x(c,1,2),   0.0   , x(c,0,2),   0.0   , x(c,1,0),   0.0   , 2*x(c,2,1),   0.0   , 2*x(c,1,2),   0.0   , 2*x(c,2,1)],
        [x(c,1,2), y(c,2,1), x(c,0,2), y(c,2,0), x(c,1,0), y(c,0,1), 2*x(c,2,1), 2*y(c,1,2), 2*x(c,1,2), 2*y(c,2,1), 2*x(c,2,1), 2*y(c,1,2)],
    ])
    B2 = (1 / 2*A) * np.array([
        [y(c,1,2),   0.0   , y(c,0,2),   0.0   , y(c,0,1),   0.0   , 2*y(c,2,0),   0.0   , 2*y(c,2,0),   0.0   , 2*y(c,0,2),   0.0   ],
        [   0.0  , x(c,2,1),   0.0   , x(c,2,0),   0.0   , x(c,1,0),   0.0   , 2*x(c,0,2),   0.0   , 2*x(c,0,2),   0.0   , 2*x(c,2,0)],
        [x(c,2,1), y(c,1,2), x(c,2,0), y(c,0,2), x(c,1,0), y(c,0,1), 2*x(c,0,2), 2*y(c,2,0), 2*x(c,0,2), 2*y(c,2,0), 2*x(c,2,0), 2*y(c,0,2)],
    ])
    B3 = (1 / 2*A) * np.array([
        [y(c,1,2),   0.0   , y(c,2,0),   0.0   , y(c,1,0),   0.0   , 2*y(c,1,0),   0.0   , 2*y(c,0,1),   0.0   , 2*y(c,0,1),   0.0   ],
        [   0.0  , x(c,2,1),   0.0   , x(c,0,2),   0.0   , x(c,0,1),   0.0   , 2*x(c,0,1),   0.0   , 2*x(c,1,0),   0.0   , 2*x(c,1,0)],
        [x(c,2,1), y(c,1,2), x(c,0,2), y(c,2,0), x(c,0,1), y(c,1,0), 2*x(c,0,1), 2*y(c,1,0), 2*x(c,1,0), 2*y(c,0,1),2*x(c,1,0), 2*y(c,0,1)],
    ])

    k = ( A * t / 3) * (B1.T@E@B1 + B2.T@E@B2 + B3.T@E@B3)

    return k
