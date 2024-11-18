# -*- coding: utf-8 -*-
"""
Vytvořeno dne 17. července 2023

@author: D. Loukrezis
"""

# %% importy

import numpy as np
import openturns as ot

from scim_model import beam_deflection
from scim_distributions import dist_beam_joint

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        'text.usetex': True,
        'figure.figsize': [6.5, 4.5],
        'figure.dpi': 300,
        'font.size': 20,
        'font.family': 'serif',
        "lines.linewidth": 3.0
    }
)
legend_font_size = 12

seed_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
nsamples = 1500
folder_str = 'beam_data_DL/'
txt_str = '.txt'
inputs_str = 'inputs'
outputs_str = 'outputs'
seed_str = '_seed='

# parametry modelu nosníku
M_values = [10, 100, 1000]  # počet bodů podél délky nosníku
Q_values = [50, 100, 150]  # počet vzorků

# Definice společné distribuce pro parametry nosníku (log-normální)
def define_beam_distribution():
    dist_list = [ot.LogNormal() for _ in range(20)]
    return ot.ComposedDistribution(dist_list)

if __name__ == '__main__':

    # iterace přes seed
    for seed in seed_list:
        ot.RandomGenerator.SetSeed(seed)
        seed_str_now = seed_str + str(seed)
        dist_joint = define_beam_distribution()
        data_in = np.array(dist_joint.getSample(nsamples))
        save_inputs = folder_str + inputs_str + seed_str_now + txt_str
        print(save_inputs)
        np.savetxt(save_inputs, data_in)
        deflections = []
        plt.figure()
        for i in range(nsamples):
            print(i)
            params = data_in[i, :5]

            deflection = beam_deflection(
                w=params[0],
                h=params[1],
                L=params[2],
                E=params[3],
                P=params[4],
                M=M_values[1],
            )
            deflections.append(deflection)
            plt.plot(np.linspace(0, params[2], M_values[1]), deflection, '-y')

        deflections = np.array(deflections)

        # Graf deformace vs. pozice
        plt.xlabel('Pozice na nosníku (m)')
        plt.ylabel('Deformace (m)')
        plt.tight_layout()
        plt.xlim(0, params[2])  # Délka nosníku
        plt.xticks(np.linspace(0, params[2], 5))

        save_outputs = folder_str + outputs_str + seed_str_now + txt_str
        print(save_outputs)
        print()
        np.savetxt(save_outputs, deflections)
