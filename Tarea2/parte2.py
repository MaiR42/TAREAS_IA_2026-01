import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
  

# Cargar dataset
secondary_mushroom = fetch_ucirepo(id=848)
X = secondary_mushroom.data.features
y = secondary_mushroom.data.targets
df = X.copy()
df["class"] = y


# Eliminar temporalmente
df = df.sample(n=15000, random_state=42).reset_index(drop=True)
print(df.shape)

####

# Eliminar columnas con missing values
df = df.drop(columns=[
    "cap-surface",
    "gill-attachment",
    "gill-spacing",
    "stem-root",
    "stem-surface",
    "veil-type",
    "veil-color",
    "ring-type",
    "spore-print-color"
])

print("df despues de eliminar columnas") # Debug
print(df.shape)
print(df.head())


# Elegir las variables objetivo
Y1 = "cap-diameter" # Continua
Y2 = "class" # Discreta

# Elegir los X
features = [
    "cap-shape",
    "cap-color",
    "does-bruise-or-bleed",
    "gill-color",
    "stem-height",
    "stem-width",
    "stem-color",
    "has-ring",
    "habitat",
    "season"
]

# Crear dataset para regresion y clasificacion


# Codificación de variables categóricas
X_enc = pd.get_dummies(df[features], drop_first=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_clf = le.fit_transform(df[Y2]) # Codificación de Y2
y_reg = df[Y1].values.astype(np.float32)

# Division entrenamiento, validacion y test

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Regre
X_train_r, X_temp_r, y_train_r, y_temp_r = train_test_split(
    X_enc,
    y_reg,
    test_size=0.30, # 70% para entrenamiento y 30% restante
    random_state=42
)
X_val_r, X_test_r, y_val_r, y_test_r = train_test_split(
    X_temp_r,
    y_temp_r,
    test_size=0.50, # 50% del 30% restante anterior -> 15% para validacion y test
    random_state=42
)
 
# Clasif
X_train_c, X_temp_c, y_train_c, y_temp_c = train_test_split(
    X_enc,
    y_clf,
    test_size=0.30, # 70% para entrenamiento y 30% restante
    random_state=42,
    stratify=y_clf
)
X_val_c, X_test_c, y_val_c, y_test_c = train_test_split(
    X_temp_c,
    y_temp_c,
    test_size=0.50, # 50% del 30% restante anterior -> 15% para validacion y test
    random_state=42,
    stratify=y_temp_c
)


