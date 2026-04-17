import pandas as pd
import numpy as np
import pgmpy as pgmpy

from pgmpy.estimators import HillClimbSearch, ExhaustiveSearch # Estos son los importantes
from pgmpy.estimators import BIC, BDeu, K2         # De estos solo se pide BIC por el moment
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


import networkx as nx # Para ver grafos
import matplotlib.pyplot as plt

import sys # Solo para mostrar version de python
import time # Solo para ver tiempo de ejecucion

### Debug
def initial_debug_time():
    global tiempo_inicio_d
    tiempo_inicio_d = time.time()

def final_debug_time():
    tiempo_final_d = time.time()
    tiempo_ejecucion_d = tiempo_final_d - tiempo_inicio_d
    print("\n**************************************************************")
    print(f"DEBUG: Tiempo de ejecucion: {tiempo_ejecucion_d:.3f} segundos")
    print("**************************************************************\n")


tiempo_inicio = time.time()

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

print("\n=== PREPROCESAMIENTO DE DATOS ===")

# Convertir a DataFrame completo (X + y)
df = X.copy()
df['poisonous'] = y.values.ravel()  # Añadir la variable objetivo

print(f"Dataset original: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Valores faltantes por columna:\n{df.isnull().sum()}")

# Eliminar columna con missing values (stalk-root)
df = df.drop(columns=['stalk-root'])
print(f"\n Quedaron {df.shape[1]} columnas en total")

print(df.info())



# # # # # # # # # # # # Construir red Bayesiana con Hill Climbing # # # # # # # # # # # # 



print("\n=== CONSTRUCCIÓN DE LA RED BAYESIANA (HILL CLIMBING) ===")

# Que hace el algoritmo de hill climbing?

# Encuentra una red bayesiana que se ajuste a los datos con una heuristica basada en un criterio de puntuacion (BIC),
# lo que implica que no garantiza un optimo global, pero encuentra un optimo local en un buen tiempo de ejecucion

initial_debug_time()

hc = HillClimbSearch(df)

modelo_hc = hc.estimate(scoring_method=BIC(df))

print("Aristas encontradas:")
print(modelo_hc.edges())

final_debug_time()

# Ver grafo
G = nx.DiGraph(modelo_hc.edges())
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42, k=2) # Posiciona los nodos de forma estética, k=2 para la separacion entre nodos
nx.draw(G, pos, with_labels=True, arrowsize=10, node_size=1000, font_size=12) 
plt.title("DAG de la Red Bayesiana Aprendida (Mushroom)", size=15)
plt.show()

# Ajustar modelo para inferencia con Maximum Likelihood Estimation (MLE)
modelo = DiscreteBayesianNetwork(modelo_hc.edges())
modelo.fit(df, estimator=MaximumLikelihoodEstimator)

# Inferencias
inferencia = VariableElimination(modelo)

i1 = inferencia.query(
    variables=['poisonous'],
    evidence={'odor': 's'} # Olor spicy¿ (picante)
)
print("\n========= INFERENCIA 1=========")
print("Probabilidad de que un hongo sea venenoso \ndado a que tiene olor picante (spicy)\n")
print("P(poisonous/odor=spicy)")
print("================================")
print(i1)

i2 = inferencia.query(
    variables=['habitat'],
    evidence={'population': 'a'} # Poblacion abundante 
)
print("\n========= INFERENCIA 2=========")
print("Probabilidad del habitad de un hongo \ndado a que su poblacion es abundante\n")
print("P(habitad/population=abundante)")
print("================================")
print(i2)


# # # # # # # # # # # # Construir red Bayesiana con Exhausive Search # # # # # # # # # # # # 

# Se redujo drasticamente el numero de columnas del dataset para que el algoritmo de Exhaustive Search
# pueda ejecutarse en un buen tiempo.

# Dejando las variables escenciales para comprobar las inferencias anteriores. 
# 5 variables por el momento (Con 6 el programa no termina de ejecutarse)
# (En total se borraron 17 columnas (22->5))

df_short = df[[
    "poisonous",
    "odor",
    "cap-color", # Deje este por si acaso, aunque no se usara en las inferencias
    "habitat",
    "population"
]]

initial_debug_time()

es = ExhaustiveSearch(df_short, scoring_method=BIC(df_short))
modelo_ex = es.estimate()

print("Aristas encontradas:")
print(modelo_ex.edges())

final_debug_time()

# Ver grafo
G = nx.DiGraph(modelo_ex.edges())
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42, k=2) # Posiciona los nodos de forma estética, k=2 para la separacion entre nodos
nx.draw(G, pos, with_labels=True, arrowsize=10, node_size=1000, font_size=12) 
plt.title("DAG de la Red Bayesiana Aprendida (Mushroom)", size=15)
plt.show()

# Ajustar modelo para inferencia con Maximum Likelihood Estimation (MLE)

modelo2 = DiscreteBayesianNetwork(modelo_ex.edges())
modelo2.fit(df, estimator=MaximumLikelihoodEstimator)

# Inferencias

inferencia = VariableElimination(modelo2)

i1 = inferencia.query(
    variables=['poisonous'],
    evidence={'odor': 's'} # Olor spicy¿ (picante)
)
print("\n========= INFERENCIA 1=========")
print("Probabilidad de que un hongo sea venenoso \ndado a que tiene olor picante (spicy)\n")
print("P(poisonous/odor=s)")
print("================================")
print(i1)

i2 = inferencia.query(
    variables=['habitat'],
    evidence={'population': 'a'} # Poblacion abundante 
)
print("\n========= INFERENCIA 2=========")
print("Probabilidad del habitad de un hongo \ndado a que su poblacion es abundante\n")
print("P(habitad/population=a)")
print("================================")
print(i2)

##################################################################################################
tiempo_final = time.time()
tiempo_ejecucion = tiempo_final - tiempo_inicio
print(f"\n\nTiempo de ejecucion: {tiempo_ejecucion:.3f} segundos\n")
