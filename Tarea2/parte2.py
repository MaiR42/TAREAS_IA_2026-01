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