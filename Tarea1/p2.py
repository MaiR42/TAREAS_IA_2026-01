import pgmpy as pgmpy

from pgmpy.estimators import HillClimbSearch, ExhaustiveSearch
from pgmpy.estimators import BIC
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling

import networkx as nx
import matplotlib.pyplot as plt

import sys # Solo para mostrar version de python
import time # Solo para ver tiempo de ejecucion

# Librerias que se colocaron ahora
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from hmmlearn import hmm

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
# Ruta base
path = "datasets/UCI HAR Dataset/"

# Cargar nombres de columnas
features = pd.read_csv(path + "features.txt", sep="\s+", header=None, names=["index", "feature"])

feature_names = features["feature"]

# hacer nombres únicos, dado a que hay columnas con nombres repetidos
feature_names = pd.Series(feature_names)
feature_names = feature_names.where(~feature_names.duplicated(), feature_names + "_" + feature_names.groupby(feature_names).cumcount().astype(str))

# Cargar TRAIN
X_train = pd.read_csv(path + "train/X_train.txt", sep="\s+", header=None, names=feature_names)
y_train = pd.read_csv(path + "train/y_train.txt", header=None, names=["activity"])
subject_train = pd.read_csv(path + "train/subject_train.txt", header=None, names=["subject"])

# Cargar TEST
X_test = pd.read_csv(path + "test/X_test.txt", sep="\s+", header=None, names=feature_names)
y_test = pd.read_csv(path + "test/y_test.txt", header=None, names=["activity"])
subject_test = pd.read_csv(path + "test/subject_test.txt", header=None, names=["subject"])

# TRAIN
train = pd.concat([subject_train, y_train, X_train], axis=1)
# TEST
test = pd.concat([subject_test, y_test, X_test], axis=1)

# Dataset completo
df = pd.concat([train, test], axis=0).reset_index(drop=True)

activity_labels = pd.read_csv(path + "activity_labels.txt", sep="\s+", header=None, names=["id", "activity_name"])

df = df.merge(activity_labels, left_on="activity", right_on="id")
df.drop("id", axis=1, inplace=True)

print("============================INFO DEL DATASET============================\n")
print("https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones \n")
print(f"Dataset con: {df.shape[0]} filas, {df.shape[1]} columnas\n")
print(df.head())
print("========================================================================\n")

# Debug: Mostrar columnas que contienen "tBodyAccMag" o "tBodyGyroMag" para saber si cambiaron de nombre por duplicados
for col in df.columns:
    if "tBodyAccMag" in col or "tBodyGyroMag" in col:
        print(col)

# # # # # A)

# Los estados ocultos son las actividades humanas (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) presentes en la
# informacion del dataset


# # # Indices para las observaciones
debug_text()

df["I_acc"] = (df["tBodyAccMag-mean()"] + df["tBodyAccMag-std()"]) / 2
df["I_gyro"] = (df["tBodyGyroMag-mean()"] + df["tBodyGyroMag-std()"]) / 2

print(df[["I_acc", "I_gyro"]].head())

# Discretizar las variables continuas en tres categorías: "bajo", "medio" y "alto"
# Criterio de discretización: se divide el rango de cada variable en tres partes iguales y se asigna una etiqueta a cada parte.
df["I_acc_disc"] = pd.qcut(
    df["I_acc"],
    q=3,
    labels=["low", "medium", "high"]
)
df["I_gyro_disc"] = pd.qcut(
    df["I_gyro"],
    q=3,
    labels=["low", "medium", "high"]
)
print(df[["I_acc_disc", "I_gyro_disc"]].head())
print(df["I_acc_disc"].value_counts())
print(df["I_gyro_disc"].value_counts())



# # # Crear Observaciones
# Las 9 observaciones posibles vienen de combinar las 3 categorias entre los indices (acc y gyro)

debug_text()

# Forzar las 9 combinaciones de observaciones, pq si no tira solo 7 combinaciones
levels = ["low", "medium", "high"]
all_obs = [f"{i}_{j}" for i in levels for j in levels]
df["observation"] = pd.Categorical(
    df["I_acc_disc"].astype(str) + "_" + df["I_gyro_disc"].astype(str),
    categories=all_obs
)
print(df["observation"].unique()) # Verificar

# # # # B)

# HMMs necesitan variables numéricas, así que se codifican las observaciones
le = LabelEncoder()
df["obs_encoded"] = le.fit_transform(df["observation"])


# Crear secuencias de observaciones
sequences_obs = []
sequences_states = []

for subject in df["subject"].unique():
    df_sub = df[df["subject"] == subject]
    
    sequences_obs.append(df_sub["obs_encoded"].values)
    sequences_states.append(df_sub["activity"].values)
print("Cantidad de secuencias:", len(sequences_obs))
print("Largo de primera secuencia:", len(sequences_obs[0]))

n_obs = len(le.classes_)
# Estimar vector inicial pi, matriz de transicion T y matriz de emision E
n_states = len(df["activity"].unique())
pi = np.zeros(n_states)

for seq in sequences_states:
    pi[seq[0] - 1] += 1  # -1 porque estados van de 1 a 6

pi = pi / pi.sum()

# Matriz de transicion T
T = np.zeros((n_states, n_states))

for seq in sequences_states:
    for i in range(len(seq) - 1):
        T[seq[i] - 1][seq[i+1] - 1] += 1

# normalizar
T = T / T.sum(axis=1, keepdims=True)

# Matriz de emision E
E = np.zeros((n_states, n_obs))

for seq_s, seq_o in zip(sequences_states, sequences_obs):
    for s, o in zip(seq_s, seq_o):
        E[s - 1][o] += 1

# normalizar
E = E / E.sum(axis=1, keepdims=True)

# # # # Crear modelo HMM
model = hmm.MultinomialHMM(n_components=6) # 6 estados ocultos (actividades)
model.startprob_ = pi
model.transmat_ = T
model.emissionprob_ = E

