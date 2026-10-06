import dill, numpy as np, os

_cesta = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_cesta, 'pce_surrogate.pkl'), 'rb') as f:
    _PCE = dill.load(f)

def model_pce(samples):
    E = np.array(samples, dtype=float).reshape(1, -1)
    return np.ravel(_PCE.predict(E))
