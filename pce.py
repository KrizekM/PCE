# -*- coding: utf-8 -*-
"""
Vytvořeno dne: Pátek 3. února 2023

Autor: D. Loukrezis
"""

import openturns as ot
import numpy as np


class PolynomialChaosExpansion():
    '''
    Třída pro flexibilní práci s polynomiálním chaosem pomocí knihovny OpenTURNS.
    Umožňuje vytvářet modely polynomiálního chaosu (PCE), analyzovat nejistoty
    a provádět citlivostní analýzu.
    '''

    def __init__(self, pdf, exp_design_in, exp_design_out):
        """
        Inicializace třídy.

        Parametry:
        - pdf: Distribuce pravděpodobnosti vstupních dat. Pokud není zadána,
          distribuce bude odvozena z experimentálních dat.
        - exp_design_in: Matice vstupních dat (experimentální design).
        - exp_design_out: Matice odpovídajících výstupních dat.
        """
        # Pokud není zadána distribuce pravděpodobnosti, odhadne se z dat
        if pdf is None:
            chaos_algo_data = ot.FunctionalChaosAlgorithm(exp_design_in, exp_design_out)
            chaos_algo_data.run()
            self.pdf = chaos_algo_data.getDistribution()
        else:
            self.pdf = pdf

        # Zjištění počtu vstupů a výstupů
        self.num_inputs = pdf.getDimension()
        self.num_outputs = exp_design_out.shape[1]
        self.num_samples = exp_design_out.shape[0]
        self.exp_design_inputs = exp_design_in
        self.exp_design_outputs = exp_design_out

        # Počáteční výpočet PCE s polynomy 1. řádu (nízká náročnost)
        self.enumerate_function = ot.LinearEnumerateFunction(self.num_inputs)

        # Výběr správných polynomů podle rozdělení vstupních dat
        self.polynomial_collection = [
            ot.StandardDistributionPolynomialFactory(self.pdf.getMarginal(i))
            for i in range(self.num_inputs)
        ]

        # Vytvoření obecné produktové báze polynomů
        self.product_basis = ot.OrthogonalProductPolynomialFactory(
            self.polynomial_collection, self.enumerate_function
        )

        # Výpočet PCE pro polynomy 1. řádu
        total_degree = 1  # Celkový stupeň PCE
        basis_size = self.enumerate_function.getStrataCumulatedCardinal(total_degree)
        adaptive_strategy = ot.FixedStrategy(self.product_basis, basis_size)
        projection_strategy = ot.LeastSquaresStrategy()
        chaos_algo = ot.FunctionalChaosAlgorithm(
            exp_design_in, exp_design_out, pdf, adaptive_strategy, projection_strategy
        )
        chaos_algo.run()
        chaos_result = chaos_algo.getResult()

        # Transformace vstupních dat pro standardizaci
        self.transformation = chaos_result.getTransformation()

        # Uložení jednotlivých a multi-indexů pro 1. řád
        self.single_index_set = np.array(chaos_result.getIndices()).tolist()
        self.multi_index_set = [
            list(self.enumerate_function(idx)) for idx in self.single_index_set
        ]

        # Uložení báze a koeficientů pro 1. řád
        self.basis = chaos_result.getReducedBasis()
        self.num_polynomials = self.basis.getSize()
        self.coefficients = np.array(chaos_result.getCoefficients())

        # Výpočet designové matice
        self.design_matrix = np.zeros([self.num_samples, self.num_polynomials])
        exp_design_inputs_tf = self.transformation(self.exp_design_inputs)
        for j in range(self.num_polynomials):
            self.design_matrix[:, j] = np.array(self.basis[j](exp_design_inputs_tf)).flatten()
