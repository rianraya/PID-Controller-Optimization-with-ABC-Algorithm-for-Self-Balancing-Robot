import numpy as np


def objective(x, u):
    cost = 0

    # =====Q & R Definition=====
    # You may set the load based on which element you want to stabilize
    Q = np.array([
        [10, 0, 0, 0],  # th
        [0, .1, 0, 0],  # th_dot
        [0, 0, 10, 0],  # phi
        [0, 0, 0, .1]   # phi_dot
    ])
    R = 0

    # =====Cost Calculation=====
    for i in range(x.shape[0]):
        cost += np.dot(np.dot(x[i].T, Q), x[i]) + np.dot(np.dot(u[i].T, R), u[i])
    return cost