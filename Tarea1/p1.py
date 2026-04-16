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
# ==================== 1. PREPROCESAMIENTO ====================
print("\n=== 1. PREPROCESAMIENTO DE DATOS ===")

# Convertir a DataFrame completo (X + y)
df = X.copy()
df['poisonous'] = y  # Añadir la variable objetivo

print(f"Dataset original: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Valores faltantes por columna:\n{df.isnull().sum()}")

# Eliminar columna con valores faltantes (stalk-root)
df_clean = df.drop(columns=['stalk-root'])
print(f"\nDespués de eliminar 'stalk-root': {df_clean.shape[1]} columnas")

# Codificar variables categóricas a números (pgmpy las necesita así)
from sklearn.preprocessing import LabelEncoder
df_encoded = df_clean.copy()
for col in df_encoded.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

print(f"Dataset listo: {df_encoded.shape[0]} filas, {df_encoded.shape[1]} columnas")
print("Ejemplo de datos codificados:\n", df_encoded.head())

# ==================== 2. APRENDIZAJE DE ESTRUCTURAS ====================
print("\n=== 2. APRENDIZAJE DE ESTRUCTURAS ===")
print("\n--- 2.1 Hill-Climbing ---")
# Explicación: Búsqueda local greedy que empieza con red vacía, añade/elimina/invierte arcos
# que maximicen una función de puntuación (BIC por defecto)
hc = HillClimbSearch(df_encoded)
hc_model = hc.estimate(scoring_method='bic')
print("Estructura aprendida (arcos):", hc_model.edges())
print(f"Número de arcos: {len(hc_model.edges())}")

print("\n--- 2.2 Búsqueda Exhaustiva ---")
# Explicación: Evalúa TODAS las posibles estructuras y selecciona la mejor.
# Con 21 columnas, hay 2^(21*20/2) ~ 2^210 posibles DAGs → imposible.
# Solución: Reducir a 5 columnas representativas para demostrar el método
columnas_reducidas = ['poisonous', 'odor', 'cap-color', 'bruises', 'gill-color']
df_reducido = df_encoded[columnas_reducidas]
print(f"Para búsqueda exhaustiva usamos {len(columnas_reducidas)} columnas: {columnas_reducidas}")
print("Justificación: El número de DAGs posibles crece super-exponencialmente. Con 5 nodos hay 29281 DAGs, manejable.")

exhaustive = ExhaustiveSearch(df_reducido, scoring_method='bic')
exhaustive_model = exhaustive.estimate()
print("Estructura exhaustiva (arcos):", exhaustive_model.edges())
print(f"Número de arcos: {len(exhaustive_model.edges())}")

# ==================== 3. ESTIMACIÓN DE PARÁMETROS ====================
print("\n=== 3. ESTIMACIÓN DE PARÁMETROS ===")
from pgmpy.estimators import BayesianEstimator

# Para Hill-Climbing (usando todo el dataset)
model_hc = BayesianNetwork(hc_model.edges())
model_hc.fit(df_encoded, estimator=BayesianEstimator, prior_type='BDeu')

# Para Exhaustiva (usando dataset reducido)
model_exh = BayesianNetwork(exhaustive_model.edges())
model_exh.fit(df_reducido, estimator=BayesianEstimator, prior_type='BDeu')

print("Parámetros estimados para ambos modelos")

# ==================== 4. INFERENCIAS (original) ====================
print("\n=== 4. INFERENCIAS A POSTERIORI (Dataset Original) ===")
from pgmpy.inference import VariableElimination

# Inferencia en modelo Hill-Climbing
infer_hc = VariableElimination(model_hc)

print("\n--- Inferencia 1 (Diagnóstico): ¿Probabilidad de venenoso dado olor a pescado? ---")
# odor=fishy (codificado como 4 según el orden original)
prob_poison_given_fishy = infer_hc.query(variables=['poisonous'], evidence={'odor': 4})
print(prob_poison_given_fishy)

print("\n--- Inferencia 2 (Diagnóstico): ¿Probabilidad de venenoso dado moretones? ---")
# bruises=t (moretones = True, codificado como 1)
prob_poison_given_bruises = infer_hc.query(variables=['poisonous'], evidence={'bruises': 1})
print(prob_poison_given_bruises)

# Inferencia en modelo Exhaustivo (con dataset reducido)
infer_exh = VariableElimination(model_exh)
print("\n--- Inferencia en modelo exhaustivo: P(venenoso | olor=pescado) ---")
prob_exh = infer_exh.query(variables=['poisonous'], evidence={'odor': 4})
print(prob_exh)

# ==================== 5. GENERACIÓN DE DATOS SINTÉTICOS ====================
print("\n=== 5. GENERACIÓN DE DATOS SINTÉTICOS ===")

def generar_y_repetir(df_original, porcentaje, modelo_referencia):
    """Genera datos sintéticos y repite el proceso"""
    n_sinteticos = int(len(df_original) * porcentaje / 100)
    print(f"\n--- Aumento {porcentaje}%: generando {n_sinteticos} filas ---")
    
    # Generar datos del modelo
    from pgmpy.sampling import BayesianModelSampling
    sampler = BayesianModelSampling(modelo_referencia)
    datos_sinteticos = sampler.forward_sample(size=n_sinteticos)
    
    # Combinar con originales
    df_completo = pd.concat([df_original, datos_sinteticos], ignore_index=True)
    print(f"  Nuevo tamaño: {len(df_completo)} filas")
    
    # Aprender estructura con Hill-Climbing
    hc_nuevo = HillClimbSearch(df_completo)
    modelo_nuevo = hc_nuevo.estimate(scoring_method='bic')
    
    # Estimar parámetros
    red_nueva = BayesianNetwork(modelo_nuevo.edges())
    red_nueva.fit(df_completo, estimator=BayesianEstimator)
    
    # Inferencia
    infer_nueva = VariableElimination(red_nueva)
    resultado = infer_nueva.query(variables=['poisonous'], evidence={'odor': 4})
    
    return {
        'tamaño': len(df_completo),
        'arcos': len(modelo_nuevo.edges()),
        'inferencia_olor_pescado': resultado
    }

# Usar modelo_hc como referencia (entrenado con datos originales)
resultados = {}
for pct in [10, 20, 40]:
    resultados[pct] = generar_y_repetir(df_encoded, pct, model_hc)

# ==================== 6. COMPARACIÓN DE RESULTADOS ====================
print("\n=== 6. COMPARACIÓN DE RESULTADOS ===")
print("Resumen de estructuras y complejidad:")
print(f"Original (Hill-Climbing): {len(model_hc.edges())} arcos")

for pct, res in resultados.items():
    print(f"Aumento {pct}%: {res['arcos']} arcos, inferencia = {res['inferencia_olor_pescado']}")

print("\n¡Proceso completado!")


