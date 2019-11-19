import numpy as np

E = {6: np.array([
        [32000000.,  8000000.,        0.],
        [ 8000000., 32000000.,        0.],
        [       0.,        0., 12000000.]
    ])
}

print(E)

Y = np.array([1, 0])
map = [6, 6]

X = Y[:] * E[map]
print(X)