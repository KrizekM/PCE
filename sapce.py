# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:22:45 2023

@author: D. Loukrezis
"""

import openturns as ot
import numpy as np
from pce import PolynomialChaosExpansion
from idx_admissibility import admissible_neighbors


class SensitivityAdaptivePCE():
    def __init__(self, pdf, exp_design_in, exp_design_out):
        # Pokud není poskytnuta PDF, automaticky se určí na základě dat
        if pdf is None:
            chaos_algo_data = ot.FunctionalChaosAlgorithm(exp_design_in, exp_design_out)
            chaos_algo_data.run()
            self.pdf = chaos_algo_data.getDistribution()  # Určení PDF z dat
        else:
            self.pdf = pdf

        self.exp_design_in = exp_design_in
        self.exp_design_out = exp_design_out

        # Inicializace PCE 1. řádu
        self.pce = PolynomialChaosExpansion(self.pdf, self.exp_design_in, self.exp_design_out)
        self.pce.compute_coefficients()

        # Aktivní multi-index set začíná pouze nulovým multi-indexem
        self.active_multi_indices = [self.pce.multi_index_set[0]]

        # Admissible multi-index set obsahuje všechny 1. řády
        self.admissible_multi_indices = self.pce.multi_index_set[1:]

        # Vypočítáme agregovanou varianci pro každý admissible multi-index
        admissible_coefficients = self.pce.coefficients[1:].tolist()
        aggregated_admissible_coefficients = np.sum(np.abs(admissible_coefficients), axis=1)

        # Najdeme multi-index s maximální agregovanou koeficienty a odebereme ho
        help_index = np.argmax(aggregated_admissible_coefficients)
        max_admissible_multi_index = self.admissible_multi_indices.pop(help_index)

        # Přidáme tento multi-index do aktivního setu
        self.active_multi_indices.append(max_admissible_multi_index)

    def construct_adaptive_basis(self, max_condition_number=1e2):
        while True:
            # Pokud je podmíněnost matice příliš vysoká, zastavíme
            if self.pce.condition_number > max_condition_number:
                break

            # Najdeme nové admissible multi-indices
            new_admissible_multi_indices = admissible_neighbors(self.active_multi_indices[-1],
                                                                self.active_multi_indices)

            # Pokud je počet termínů větší než počet tréninkových dat, zastavíme
            num_terms = len(self.active_multi_indices) + len(self.admissible_multi_indices) + len(
                new_admissible_multi_indices)
            if num_terms >= len(self.pce.exp_design_inputs):
                break

            # Aktualizujeme admissible multi-index set
            self.admissible_multi_indices += new_admissible_multi_indices

            # Vypočítáme PCE pro plný multi-index set
            all_multi_indices = self.active_multi_indices + self.admissible_multi_indices
            self.pce.set_multi_index_set(all_multi_indices)
            self.pce.construct_basis()
            self.pce.compute_coefficients()

            # Počítáme agregované koeficienty pro každý admissible multi-index
            idx = len(self.active_multi_indices)
            admissible_coefficients = self.pce.coefficients[idx:].tolist()
            aggregated_admissible_coefficients = np.sum(np.abs(admissible_coefficients), axis=1)

            # Najdeme multi-index s maximální agregovanou variancí a přidáme ho do aktivního setu
            help_index = np.argmax(aggregated_admissible_coefficients)
            max_admissible_multi_index = self.admissible_multi_indices.pop(help_index)
            self.active_multi_indices.append(max_admissible_multi_index)

    def construct_active_pce(self):
        # Vytvoření PCE s aktivním multi-index setem
        pce = PolynomialChaosExpansion(self.pdf, self.exp_design_in, self.exp_design_out)
        pce.set_multi_index_set(self.active_multi_indices)
        pce.construct_basis()
        pce.compute_coefficients()
        return pce

    def construct_augmented_pce(self):
        # Vytvoření PCE s rozšířeným multi-index setem
        pce = PolynomialChaosExpansion(self.pdf, self.exp_design_in, self.exp_design_out)
        pce.set_multi_index_set(self.active_multi_indices + self.admissible_multi_indices)
        pce.construct_basis()
        pce.compute_coefficients()
        return pce

    def construct_reduced_augmented_pce(self, max_condition_number=1e2):
        # Vytvoříme augmented PCE
        pce = self.construct_augmented_pce()

        while True:
            # Pokud je podmíněnost matice v pořádku, nebo je počet termínů menší než vstupy, zastavíme
            if pce.condition_number <= max_condition_number and len(pce.multi_index_set) <= len(pce.exp_design_inputs):
                break

            # Odebereme multi-index s nejmenšími koeficienty
            idx_min = np.argmin(np.sum(np.abs(pce.coefficients), axis=1))
            pce.multi_index_set.pop(idx_min)
            pce.single_index_set.pop(idx_min)

            # Znovu spočítáme PCE s redukovaným multi-index setem
            pce.construct_basis()
            pce.compute_coefficients()

        return pce
