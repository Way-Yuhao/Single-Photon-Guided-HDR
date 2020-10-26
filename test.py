import numpy as np
A = np.array([[100, 200], [300, 40000]])
for p in np.nditer(A, op_flags=['readwrite']):
    p[...] = np.random.poisson(p)

print(A)