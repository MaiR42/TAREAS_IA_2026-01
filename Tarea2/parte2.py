import pandas as pd
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
secondary_mushroom = fetch_ucirepo(id=848) 
  
# data (as pandas dataframes) 
X = secondary_mushroom.data.features 
y = secondary_mushroom.data.targets 

df = X.copy()
df["class"] = y

#### Temporal, quitar filas

df = df.sample(
    n=15000,
    random_state=42
).reset_index(drop=True)

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

print("df despues de eliminar columnas")
print(df.shape)
print(df.head())


# Elegir las Y
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

X = df[features]
X = pd.get_dummies(X, drop_first=True)

y_reg = df[Y1]  # Y1
y_clf = df[Y2]  # Y2

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y_clf = le.fit_transform(y_clf)

# Etapa de entrenamiento (Regresion)

from sklearn.model_selection import train_test_split

X_train, X_temp, y_reg_train, y_reg_temp = train_test_split(
    X,
    y_reg,
    test_size=0.30, # 70% para entrenamiento y 30% restante
    random_state=42
)

X_val, X_test, y_reg_val, y_reg_test = train_test_split(
    X_temp,
    y_reg_temp,
    test_size=0.50, # 50% del 30% restante anterior -> 15% para validacion y test
    random_state=42
)

# Etapa de entrenamiento (Clasificacion)

X_train2, X_temp2, y_clf_train, y_clf_temp = train_test_split(
    X,
    y_clf,
    test_size=0.30,
    random_state=42,
    stratify=y_clf
)

X_val2, X_test2, y_clf_val, y_clf_test = train_test_split(
    X_temp2,
    y_clf_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_clf_temp
)

