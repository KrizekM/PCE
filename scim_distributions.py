# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:20:29 2023

@author: Krizek M
"""
import numpy as np
import openturns as ot
# Funkce pro převod střední hodnoty a směrodatné odchylky na parametry lognormálního rozdělení
def calculate_lognormal_params(mean, std_dev):
    sigma = np.sqrt(np.log(1 + (std_dev / mean) ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return mu, sigma

# Parametry lognormálních rozdělení
params = {
    "w": (0.15, 0.0075),         # šířka
    "h": (0.3, 0.015),          # výška
    "L": (5.0, 0.05),           # délka
    "E": (3e10, 4.5e9),         # Youngův modul pružnosti
    "P": (1e4, 2e3),            # zatížení
    "d1_d15": (10.0, 1.0)       # dummy proměnné
}
dist_w = ot.LogNormalMuSigma(*calculate_lognormal_params(*params["w"]))
dist_h = ot.LogNormalMuSigma(*calculate_lognormal_params(*params["h"]))
dist_L = ot.LogNormalMuSigma(*calculate_lognormal_params(*params["L"]))
dist_E = ot.LogNormalMuSigma(*calculate_lognormal_params(*params["E"]))
dist_P = ot.LogNormalMuSigma(*calculate_lognormal_params(*params["P"]))
dist_dummy = ot.LogNormalMuSigma(*calculate_lognormal_params(*params["d1_d15"]))

# Seskupení dummy proměnných do pole rozdělení
dummy_distributions = [dist_dummy] * 15  # 15 dummy proměnných

# Společné rozdělení všech veličin
dist_joint = ot.ComposedDistribution(
    [dist_w, dist_h, dist_L, dist_E, dist_P] + dummy_distributions
)
print("Společné rozdělení:")
print(dist_joint)
