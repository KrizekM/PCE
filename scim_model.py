import numpy as np
import openturns as ot
import matplotlib.pyplot as plt

# Vypnutí LaTeXu pro texty
plt.rcParams.update({'text.usetex': False})

# Nastavení základních parametrů, vykreslení grafu
plt.rcParams.update({
    'figure.figsize': [6, 4],
    'figure.dpi': 200,
    'font.size': 16,
})

# Funkce pro výpočet průhybu nosníku
def beam_deflection(w, h, L, E, P, lm):
    """
    Výpočet průhybu nosníku na základě zadaných parametrů.

    Parametry:
        w (float): šířka nosníku
        h (float): výška nosníku
        L (float): délka nosníku
        E (float): modul pružnosti
        P (float): rovnoměrné zatížení
        lm (array): souřadnice podél délky nosníku (numpy array)

    Návratová hodnota:
        array: průhyb nosníku pro každý bod lm
    """
    lm = np.array(lm)
    w = float(w[0]) if isinstance(w, ot.Point) else float(w)
    h = float(h[0]) if isinstance(h, ot.Point) else float(h)
    L = float(L[0]) if isinstance(L, ot.Point) else float(L)
    E = float(E[0]) if isinstance(E, ot.Point) else float(E)
    P = float(P[0]) if isinstance(P, ot.Point) else float(P)
    lm = np.array(lm, dtype=float)  #

    deflections = P * lm * ((L ** 3) - (2 * lm ** 2 * L) + (lm ** 3)) / (2 * E * w * h ** 3)

    deflections[0] = 0
    deflections[-1] = 0
    return deflections

# Parametry log-normálního rozdělení (střední hodnota a směrodatná odchylka)
mean_values = {
    'w': 0.15,  # šířka nosníku [m]
    'h': 0.3,   # výška nosníku [m]
    'L': 5.0,   # délka nosníku [m]
    'E': 3e10,  # modul pružnosti [Pa]
    'P': 1e4,   # rovnoměrné zatížení [N/m]
}

std_dev_values = {
    'w': 0.0075,
    'h': 0.015,
    'L': 0.05,
    'E': 4.5e9,  # Hodnota pro E zůstává
    'P': 2e3,
}

# Log-normální rozdělení pro parametry nosníku
distributions = [
    ot.LogNormal(np.log(mean_values['w']), std_dev_values['w']),
    ot.LogNormal(np.log(mean_values['h']), std_dev_values['h']),
    ot.LogNormal(np.log(mean_values['L']), std_dev_values['L']),
    ot.LogNormal(np.log(mean_values['E']) - 0.5 * (std_dev_values['E'] / mean_values['E'])**2,std_dev_values['E'] / mean_values['E']),
    ot.LogNormal(np.log(mean_values['P']) - 0.5 * (std_dev_values['P'] / mean_values['P'])**2,std_dev_values['P'] / mean_values['P'])
]

# Dummy parametry (15 dalších rozměrů)
for _ in range(15):
    distributions.append(ot.LogNormal(np.log(10.0), 1.0))  # Dummy parametry

# Společné rozdělení vstupů
joint_distribution = ot.ComposedDistribution(distributions)

# Nastavení bodů podél nosníku
M_list = [10, 100, 1000]  # Počet bodů pro průhyb
Q_list = [50, 100, 150]  # Počet tréninkových vzorků

# Výpočty pro každý počet bodů M a počet vzorků Q
for M in M_list:
    lm = np.linspace(0, mean_values['L'], M + 2)[1:-1]  # Souřadnice podél délky nosníku
    for Q in Q_list:
        print(f"Simulace: M = {M}, Q = {Q}")

        # Generování náhodných vstupů
        samples = joint_distribution.getSample(Q)
        w, h, L, E, P = samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3], samples[:, 4]
        deflections = np.zeros((Q, M))
        for i in range(Q):
            deflections[i, :] = beam_deflection(w[i], h[i], L[i], E[i], P[i], lm)

        plt.figure()
        for i in range(Q):
            plt.plot(lm, deflections[i, :], label=f'Vzorek {i + 1}' if i < 3 else None)
        plt.xlabel('Délka nosníku (m)')
        plt.ylabel('Průhyb (m)')
        plt.title(f'Průhyb nosníku ($M={M}, Q={Q}$)')
        plt.grid()
        if Q <= 3:
            plt.legend()
        plt.tight_layout()
        plt.show()
