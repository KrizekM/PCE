#  musi byt samostatny .py soubor
import numpy as np, time

L, I_MOM = 2.0, 4.0e-6
F_GRID = np.array([1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.])
NAROCNOST_MODELU = 0.02  # [s] - umele zpomaleni, jako by slo o drahou simulaci

def model_prehybu(samples):
    time.sleep(NAROCNOST_MODELU)
    E = float(samples[0])
    return F_GRID * L**3 / (48.0 * E * I_MOM)
