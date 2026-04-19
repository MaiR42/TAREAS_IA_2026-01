import pandas as pd
import numpy as np
import pgmpy as pgmpy

from pgmpy.estimators import HillClimbSearch, ExhaustiveSearch
from pgmpy.estimators import BIC
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from ucimlrepo import fetch_ucirepo

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

def debug_text(): # Pa ver en que parte del codigo estoy,,, tmb lo voy a usar para cuando cambio de tema
    global n_debug
    print("\n****************************************************************************************************************************")
    print("DEBUG: ", n_debug)
    print("****************************************************************************************************************************\n")
    n_debug += 1
n_debug = 0

tiempo_inicio = time.time()

print("===============================VERSIONES================================")
print("python version:", sys.version)
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("pgmpy version:", pgmpy.__version__)
print("========================================================================")


# Cargar el dataset
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

# Va buscando soluciones de manera iterativa, evaluando el movimiento de nodos en base a un criterio de puntuación (BIC),
#  lo que implica que no garantiza un óptimo global, pero encuentra un óptimo local en un tiempo de ejecución razonable


hc = HillClimbSearch(df)

modelo_hc = hc.estimate(scoring_method=BIC(df))

print("Aristas encontradas:")
print(modelo_hc.edges())


# Ver grafo
G = nx.DiGraph(modelo_hc.edges())
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42, k=2) # Posiciona los nodos de forma estética, k=2 para la separacion entre nodos
nx.draw(G, pos, with_labels=True, arrowsize=10, node_size=1000, font_size=12) 
plt.title("DAG de la Red Bayesiana Aprendida (Mushroom)", size=15, color='blue')
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
print("Probabilidad del habitad de un hongo \ndado a que su poblacion es abundante (abundant)\n")
print("P(habitad/population=abundante)")
print("================================")
print(i2)


# # # # # # # # # # # # Construir red Bayesiana con Exhausive Search # # # # # # # # # # # # 

# Explora todas las soluciones posibles del problema, las cuales luego son evaluadas con un sistema de puntuacion (BIC),
# el hecho de que explore todo implica que encuentra un optimo global,
# pero en un tiempo de ejecucion que crece de manera exponencial en relacion a la cantidad de nodos.

# Se dejaron las variables escenciales para comprobar las inferencias anteriores. 
# 5 variables por el momento (Con 6 el programa no termina de ejecutarse)
# (En total se borraron 17 columnas (22->5))

df_short = df[[
    "poisonous",
    "odor",
    "cap-color", # Deje este por si acaso, aunque no se usara en las inferencias
    "habitat",
    "population"
]]


es = ExhaustiveSearch(df_short, scoring_method=BIC(df_short))
modelo_ex = es.estimate()

print("Aristas encontradas:")
print(modelo_ex.edges())


# Ver grafo
G = nx.DiGraph(modelo_ex.edges())
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42, k=2) # Posiciona los nodos de forma estética, k=2 para la separacion entre nodos
nx.draw(G, pos, with_labels=True, arrowsize=10, node_size=1000, font_size=12)
plt.title("DAG de la Red Bayesiana Aprendida (Mushroom)", size=15)
plt.show()

# Ajustar modelo para inferencia con Maximum Likelihood Estimation (MLE)

modelo2 = DiscreteBayesianNetwork(modelo_ex.edges())
modelo2.fit(df_short, estimator=MaximumLikelihoodEstimator)

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
print("Probabilidad del habitad de un hongo \ndado a que su poblacion es abundante (abundant)\n")
print("P(habitad/population=a)")
print("================================")
print(i2)

# # # # # # # # # # # # # # # # # # # # # # # # Obtener datos sinteticos # # # # # # # # # # # # # # # # # # # # # # # # 

# A partir del modelo con hill climbing
sampler = BayesianModelSampling(modelo)

n = len(df)

data_10 = sampler.forward_sample(size=int(n * 0.10))
df_10 = pd.concat([df, data_10], ignore_index=True)
print(f"Dataset con 10% de datos sinteticos: {df_10.shape[0]} filas, {df_10.shape[1]} columnas")

data_20 = sampler.forward_sample(size=int(n * 0.20))
df_20 = pd.concat([df, data_20], ignore_index=True)
print(f"Dataset con 20% de datos sinteticos: {df_20.shape[0]} filas, {df_20.shape[1]} columnas")

data_40 = sampler.forward_sample(size=int(n * 0.40))
df_40 = pd.concat([df, data_40], ignore_index=True)
print(f"Dataset con 40% de datos sinteticos: {df_40.shape[0]} filas, {df_40.shape[1]} columnas")

# # # # # # # # # Volver a construir la red bayesiana con los nuevos datasets # # # # # # # # #

# Hill Climb

hc_10 = HillClimbSearch(df_10)
modelo_hc_10 = hc_10.estimate(scoring_method=BIC(df_10))
print("Aristas encontradas (df_10):")
print(modelo_hc_10.edges())
modelo_10 = DiscreteBayesianNetwork(modelo_hc_10.edges())
modelo_10.fit(df_10, estimator=MaximumLikelihoodEstimator)

hc_20 = HillClimbSearch(df_20)
modelo_hc_20 = hc_20.estimate(scoring_method=BIC(df_20))
print("Aristas encontradas (df_20):")
print(modelo_hc_20.edges())
modelo_20 = DiscreteBayesianNetwork(modelo_hc_20.edges())
modelo_20.fit(df_20, estimator=MaximumLikelihoodEstimator)

hc_40 = HillClimbSearch(df_40)
modelo_hc_40 = hc_40.estimate(scoring_method=BIC(df_40))
print("Aristas encontradas (df_40):")
print(modelo_hc_40.edges())
modelo_40 = DiscreteBayesianNetwork(modelo_hc_40.edges())
modelo_40.fit(df_40, estimator=MaximumLikelihoodEstimator)

for i in range(3):
    print(f"\n========= INFERENCIA 1 (df_{[10,20,40][i]}) =========")
    inferencia = VariableElimination([modelo_10, modelo_20, modelo_40][i])
    i1 = inferencia.query(
        variables=['poisonous'],
        evidence={'odor': 's'} # Olor spicy¿ (picante)
    )
    print("Probabilidad de que un hongo sea venenoso \ndado a que tiene olor picante (spicy)\n")
    print(f"P(poisonous/odor=s) con df_{[10,20,40][i]}")
    print("================================")
    print(i1)
    
    print(f"\n========= INFERENCIA 2 (df_{[10,20,40][i]}) =========")
    i2 = inferencia.query(
    variables=['habitat'],
    evidence={'population': 'a'} # Poblacion abundante 
    )
    print("Probabilidad del habitad de un hongo \ndado a que su poblacion es abundante (abundant)\n")
    print(f"P(habitad/population=a) con df_{[10,20,40][i]}")
    print("================================")
    print(i2)



# Exhaustive Search

cols = ["poisonous", "odor", "cap-color", "habitat", "population"]

df_10_short = df_10[cols]
df_20_short = df_20[cols]
df_40_short = df_40[cols]

es_10 = ExhaustiveSearch(df_10_short, scoring_method=BIC(df_10_short))
modelo_ex_10 = es_10.estimate()
print("Aristas encontradas:")
print(modelo_ex_10.edges())
modelo2_10 = DiscreteBayesianNetwork(modelo_ex_10.edges())
modelo2_10.fit(df_10, estimator=MaximumLikelihoodEstimator)

es_20 = ExhaustiveSearch(df_20_short, scoring_method=BIC(df_20_short))
modelo_ex_20 = es_20.estimate()
print("Aristas encontradas:")
print(modelo_ex_20.edges())
modelo2_20 = DiscreteBayesianNetwork(modelo_ex_20.edges())
modelo2_20.fit(df_20, estimator=MaximumLikelihoodEstimator)

es_40 = ExhaustiveSearch(df_40_short, scoring_method=BIC(df_40_short))
modelo_ex_40 = es_40.estimate()
print("Aristas encontradas:")
print(modelo_ex_40.edges())
modelo2_40 = DiscreteBayesianNetwork(modelo_ex_40.edges())
modelo2_40.fit(df_40, estimator=MaximumLikelihoodEstimator)

for i in range(3):
    print(f"\n========= INFERENCIA 1 (df_{[10,20,40][i]}_short) =========")
    inferencia = VariableElimination([modelo2_10, modelo2_20, modelo2_40][i])
    i1 = inferencia.query(
        variables=['poisonous'],
        evidence={'odor': 's'} # Olor spicy¿ (picante)
    )
    print("Probabilidad de que un hongo sea venenoso \ndado a que tiene olor picante (spicy)\n")
    print(f"P(poisonous/odor=s) con df_{[10,20,40][i]}_short")
    print("================================")
    print(i1)
    
    print(f"\n========= INFERENCIA 2 (df_{[10,20,40][i]}_short) =========")
    i2 = inferencia.query(
    variables=['habitat'],
    evidence={'population': 'a'} # Poblacion abundante 
    )
    print("Probabilidad del habitad de un hongo \ndado a que su poblacion es abundante (abundant)\n")
    print(f"P(habitad/population=a) con df_{[10,20,40][i]}_short")
    print("================================")
    print(i2)

##################################################################################################
tiempo_final = time.time()
tiempo_ejecucion = tiempo_final - tiempo_inicio
print(f"\n\nTiempo de ejecucion: {tiempo_ejecucion:.3f} segundos\n")
