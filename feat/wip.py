
x = lambda a, i, j: a[i][0] - a[j][0]
y = lambda b, i, j: b[i][1] - b[j][1]

X = lambda c, e, i, j: c[e[:,i]][:,0] - c[e[:,j]][:,0]
Y = lambda c, e, i, j: c[e[:,i]][:,1] - c[e[:,j]][:,1]


def compute_K_entry(row, col, c, e, E_array, t):
    J = X(c,e,1,0) * Y(c,e,2,0) - X(c,e,2,0) * Y(c,e,1,0)

    b = np.array([
        [Y(c,e,1,2), Y(c,e,2,0), Y(c,e,0,1)],
        [X(c,e,2,1), X(c,e,0,2), X(c,e,1,0)],
    ])

    A = row % 2
    B = row + (-row // 2)
    C = col % 2
    D = col + (-col // 2)

    E = int(row % 2 == 0)
    F = (row + (-1)**row) + (-(row + (-1)**row)//2)
    G = int(col % 2 == 0)
    H = (col + (-1)**col) + (-(col + (-1)**col)//2)

    k_data = (b[A,B] * b[C,D] * E_array[:,(row+col) % 2] + b[E,F] * b[G,H] * E_array[:,5]) / J * t * 0.5
    return k_data