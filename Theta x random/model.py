import numpy as np

def h(theta, predict, i, x0):
    x = x0.copy()
    x[i] = theta[0, 0]
    return np.asarray(predict(x[None]), dtype=float).ravel()
