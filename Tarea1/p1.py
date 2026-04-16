import pandas as pd
import numpy as np
import pgmpy as pgmpy
from pgmpy.estimators import HillClimbSearch, ExhaustiveSearch # Estos son los importantes
from pgmpy.estimators import BIC, BDeu, K2         # Estos creo q no los piden ***quisas borrar despues***
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator # Tampoco se pide ***quisas borrar despues***
import sys # Solo para mostrar version de python

print("===============================VERSIONES================================")
print("python version:", sys.version)
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("pgmpy version:", pgmpy.__version__)
print("========================================================================")


# Cargar el dataset

from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=73) # Dataset: Mushroom
X = dataset.data.features 
y = dataset.data.targets 

print("============================INFO DEL DATASET============================\n")
print("https://archive.ics.uci.edu/dataset/73/mushroom \n")
print(f"El dataset tiene {X.shape[0]} filas y {X.shape[1]} columnas")
print(X.isnull().sum().sum(), "valores faltantes en total\n") # Quisas despues borrar la columna con valores faltantes

print(dataset.variables)
print("========================================================================\n")
############################################################


