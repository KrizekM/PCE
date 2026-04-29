import itertools
import math
import numpy as np
import UQpy
from scipy.special import comb
from UQpy.distributions import Uniform, JointIndependent
from UQpy.surrogates import *
from UQpy.surrogates.polynomial_chaos import TotalDegreeBasis, LeastSquareRegression
from UQpy.sensitivity import PceSensitivity
from UQpy.sampling import MonteCarloSampling
from sklearn.metrics import mean_squared_error, r2_score
import time  
import scipy as sp



def is_admissible(index, index_set):
    """
    Zjistí, zda je daný multi-index přípustný
    (všechny jeho zpětné sousedy jsou v index_set)
    
    Parametry:
    index : np.array - testovaný multi-index
    index_set : list - seznam multi-indexů v aktuální bázi
    """
    back_neighbors = backward_neighbors(index)  # Získá zpětné sousedy
    
    for ind_b in back_neighbors:
        # Oprava: porovnání polí po prvcích
        if not any(np.array_equal(ind_b, x) for x in index_set):
            return False
    return True

def admissible_neighbors(index, index_set):
    """
    Najde všechny přípustné sousedy daného indexu
    
    Parametry:
    index : np.array - výchozí multi-index
    index_set : list - seznam povolených indexů
    """
    for_neighbors = forward_neighbors(index)  # Získá dopředné sousedy
    
    # Filtruje jen přípustné sousedy
    for_truefalse = [is_admissible(fn, index_set) for fn in for_neighbors]
    adm_neighbors = np.array(for_neighbors)[for_truefalse].tolist()
    
    return adm_neighbors


def forward_neighbors(index):
    """Vrátí dopředné sousedy - každou dimenzi zvýší o 1 (např. (2,1) → (3,1), (2,2))"""
    N = len(index)  # Počet dimenzí multiindexu
    for_neighbors = []
    for i in range(N):  # Pro každou dimenzi
        index_tmp = index[:]  # Kopie původního indexu
        index_tmp[i] = index_tmp[i] + 1  # Inkrementace aktuální dimenze
        for_neighbors.append(index_tmp)  # Přidání do seznamu sousedů
    return for_neighbors


def backward_neighbors(index):
    """Vrátí zpětné sousedy - každou dimenzi sníží o 1 (pokud možno, např. (2,2) → (1,2), (2,1))"""
    N = len(index)  # Počet dimenzí multiindexu
    back_neighbors = []
    for i in range(N):  # Pro každou dimenzi
        index_tmp = index[:]  # Kopie původního indexu
        if index_tmp[i] > 0:  # Pokud lze snížit (nenulová hodnota)
            index_tmp[i] = index_tmp[i] - 1  # Dekrementace aktuální dimenze
            back_neighbors.append(index_tmp)  # Přidání do seznamu sousedů
    return back_neighbors

# -*- coding: utf-8 -*-
# import openturns as ot
import numpy as np
import scipy as sp


from UQpy.distributions import Normal, JointIndependent
from UQpy.surrogates.polynomial_chaos import TotalDegreeBasis

from UQpy.surrogates.polynomial_chaos.regressions import LeastSquareRegression
#Generuje multiindexy 1. stupně - PCE

def generate_first_degree_indices(num_inputs):
    return np.eye(num_inputs, dtype=int).tolist()
    
class SensitivityAdaptivePCE:
    
    
    
            
            
            
    def __init__(self, pdf, exp_design_in, exp_design_out, num_inputs, max_partial_degree=10):
        self.pdf = pdf if pdf is not None else JointIndependent([Uniform(0, 1) for _ in range(exp_design_in.shape[1])])
        self.exp_design_in = exp_design_in
        self.exp_design_out = exp_design_out
        self.max_partial_degree = max_partial_degree
        self.num_inputs=num_inputs
        
        
    



        # #Vytvoří množinu multiindexů pro polynomy 1. stupně
        # td1_set = globals()['generate_first_degree_indices'](num_inputs)

        self.pce = PolynomialChaosExpansion(polynomial_basis=TotalDegreeBasis(pdf, 1), regression_method=LeastSquareRegression())
        self.pce.set_data(self.exp_design_in, self.exp_design_out)
        
        #Nastaví množinu multiindexů definovanou v  td1_set
        #self.pce.set_multi_index_set(td1_set) 
        
        
        #Neni potreba
        #self.pce.construct_basis() #Sestaví ortogonální polynomiální bázi na základě
        
        #Spočítá koeficienty PCE metodou nejmenších čtverců
        self.compute_coefficients()
        

        #Začíná s nulovým multi-indexem
        self.active_multi_indices =  [self.pce.multi_index_set[0]]
        self.admissible_multi_indices =  self.pce.multi_index_set[1:].tolist() #Inicializace přípustných multi-indexůObsahuje všechny multi-indexy 1. stupně
        admissible_coefficients = self.pce.coefficients[1:].tolist() # Pro každý přípustný multi-index spočítá součet absolutních hodnot jeho koeficientů
        aggregated_admissible_coefficients = np.sum(np.abs(admissible_coefficients), axis=1) #absolutni hodnota, bez ohledu na zanemenko
        help_index = np.argmax(aggregated_admissible_coefficients)
        
        # Odstraní a uloží daný multi-index z admissible množiny
        max_admissible_multi_index = self.admissible_multi_indices.pop(help_index)
        # Přidá do aktivní množiny multi-indexů
        self.active_multi_indices.append(max_admissible_multi_index)

        # Odstraní odpovídající hodnotu z agregovaných koeficientů
        # aggregated_admissible_coefficients = np.delete(aggregated_admissible_coefficients, help_index)
        
        
    def compute_coefficients(self):
        self.pce.coefficients, _, _, sv = np.linalg.lstsq(
        self.pce.design_matrix, 
        self.pce.experimental_design_output, 
            rcond=None)
    
        self.condition_number = sv[0]/sv[-1]
        
                
        
    def set_multi_index_set(self,multi_index_set):
        self.pce.multi_index_set = multi_index_set
        
        self.pce.polynomial_basis.polynomials_number = len(self.pce.multi_index_set)
        
        
        self.pce.polynomial_basis.polynomials = polynomial_chaos.polynomials.baseclass.PolynomialBasis.construct_arbitrary_basis(self.num_inputs, self.pdf, self.pce.multi_index_set)
        self.pce.set_data(self.exp_design_in, self.exp_design_out)
        
    def construct_adaptive_basis(self, max_condition_number=1e2, termination_info=True):
        

        while True:
            # T1: Kontroluje, zda číslo podmíněnosti aktuální matice překročilo limit
            if self.condition_number > max_condition_number:
                if termination_info:
                    print("Adaptive basis construction terminated:" 
                        + " design matrix not sufficiently well-conditioned:",self.condition_number)
                break
                
                
            

            # Hledání nových přípustných multi-indexů
            new_admissible_multi_indices = admissible_neighbors(
                                            self.active_multi_indices[-1],
                                            self.active_multi_indices)
            #print("new_admissible_multi_indices")
            #print("\033[95m" + str(new_admissible_multi_indices) + "\033[0m")
            # T2: Zajišťuje, že všechny nové přípustné multi-indexy splňují podmínku maximálního parciálního stupně, Stupně polynomu
            
            for idx, adm_multi_Indcs in reversed(list(enumerate(new_admissible_multi_indices))):
                for adm_multi_Indx in adm_multi_Indcs:
                    if adm_multi_Indx > self.max_partial_degree:
                        new_admissible_multi_indices.pop(idx)
                            
            #Kontroluje, zda číslo podmíněnosti aktuální matice překročilo limit
            # if [self.max_partial_degree]*self.num_inputs in self.active_multi_indices: 
            #     target_index = np.array([self.max_partial_degree] * self.num_inputs)
            #     index_found = any(np.array_equal(target_index, idx) for idx in self.active_multi_indices)
            #     if index_found: 
            #         if len(new_admissible_multi_indices) == 0:
            #             if termination_info:
            #                 print("Adaptive basis construction terminated:" 
            #                 + " maximum partial degree reached.")
            #             break
            # Kontroluje, zda počet členů báze nepřekročil počet trénovacích vzorků
            num_terms = len(self.active_multi_indices) + \
                        len(self.admissible_multi_indices) +\
                        len(new_admissible_multi_indices)
            #print("num_terms")
           # print("\033[95m" + str(num_terms) + "\033[0m")
            
            if num_terms >= len(self.exp_design_in):
                if termination_info:
                    print("Adaptive basis construction terminated:" 
                        + " basis cardinality reached experimental design size.")
                break
             
            
            self.admissible_multi_indices += new_admissible_multi_indices  #Přidá nově nalezené přípustné multi-indexy do stávající množiny přípustných indexů
            
            all_multi_indices = self.active_multi_indices + \
                                                self.admissible_multi_indices

            self.set_multi_index_set(all_multi_indices) #Nastaví kompletní množinu indexů do PCE modelu
            self.compute_coefficients()
            
            
            idx = len(self.active_multi_indices)

            admissible_coefficients = self.pce.coefficients[idx:].tolist()

            admissible_coefficients_array = np.array(admissible_coefficients)  # Convert to numpy array first
            if len(admissible_coefficients_array.shape) == 1:
                aggregated_admissible_coefficients = np.sum(np.abs(admissible_coefficients))
    
            else:
    
                aggregated_admissible_coefficients = np.sum(np.abs(admissible_coefficients), axis=1)
            
            help_index = np.argmax(aggregated_admissible_coefficients)
            max_admissible_multi_index = self.admissible_multi_indices.pop(help_index)
            self.active_multi_indices.append(max_admissible_multi_index)
            
            
            
    def construct_active_pce(self):      ## Zacit zde

        self.set_multi_index_set(self.active_multi_indices)
        self.compute_coefficients()
  
    
    # nový polynomiální chaosový expanzní (PCE) model, který kombinuje aktuální aktivní i přípustné multi-indexy
    # Není to jen "srovnání" – přímo ovlivňuje sestavu týmu!
    def construct_augmented_pce(self):

        self.set_multi_index_set(self.active_multi_indices + 
                            self.admissible_multi_indices)
        self.compute_coefficients()
        
    def construct_pruned_pce(self, cr=1e-8):
        # compute augmented pce
        pce = self.construct_augmented_pce()        
        multindex_set=self.pce.multi_index_set
        
        for i, elem in reversed(list(enumerate(np.sum(self.pce.coefficients,axis=1)))):
            if np.abs(elem) < cr:
                multindex_set.pop(i)

        # update pce multi indices and single indices
        self.set_multi_index_set(multindex_set)
        # we need to construct a new basis to update 
        # the associated paramters in the pce object.
        # If one does not construct a new basis the design matrix
        # will be built up on the old, "unpruned" basis size.
        # ->    So we basically update the basis and 
        #       the number of basis polynomials used.
        self.compute_coefficients()
        
        return pce
   
