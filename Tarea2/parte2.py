import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import matplotlib.pyplot as plt

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

### Crear dataset para regresion y clasificacion

# Codificación de variables categóricas
X_enc = pd.get_dummies(df[features], drop_first=True)

from sklearn.preprocessing import LabelEncoder

# Codificación de Y2 p → 0 (venenoso), e → 1 (comestible)
le = LabelEncoder()
y_clf = le.fit_transform(df[Y2]) 
y_reg = df[Y1].values.astype(np.float32)


## Division ENTRENAMIENTO, VALIDACION y TEST

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Regresion
X_tr_r, X_tmp_r, y_tr_r, y_tmp_r = train_test_split(X_enc, y_reg, test_size=0.30, random_state=42) # 70% para entrenamiento y 30% restante
X_val_r, X_te_r, y_val_r, y_te_r = train_test_split(X_tmp_r, y_tmp_r, test_size=0.50, random_state=42) # 50% del 30% restante anterior -> 15% para validacion y test

# Clasificacion
X_tr_c, X_tmp_c, y_tr_c, y_tmp_c = train_test_split(X_enc, y_clf, test_size=0.30, random_state=42, stratify=y_clf) # 70% para entrenamiento y 30% restante
X_val_c, X_te_c, y_val_c, y_te_c = train_test_split(X_tmp_c, y_tmp_c, test_size=0.50, random_state=42, stratify=y_tmp_c) # 50% del 30% restante anterior -> 15% para validacion y test

# Normalizacion (StandardScaler ajustado solo para train)
scaler = StandardScaler()
X_tr_r  = scaler.fit_transform(X_tr_r)
X_val_r = scaler.transform(X_val_r)
X_te_r  = scaler.transform(X_te_r)

X_tr_c  = scaler.fit_transform(X_tr_c)
X_val_c = scaler.transform(X_val_c)
X_te_c  = scaler.transform(X_te_c)

print(f"Train: {X_tr_r.shape[0]} | Val: {X_val_r.shape[0]} | Test: {X_te_r.shape[0]}")
print(f"Número de features tras one-hot: {X_tr_r.shape[1]}")

### Otros optimizadores 

### FUNCIONES DE PERDIDA Y GRADIENTES

def mse_grad(X, y, w):
    n = len(y)
    pred = X @ w
    return (2 / n) * X.T @ (pred - y)

def bce_grad(X, y, w):
    n = len(y)
    pred = 1 / (1 + np.exp(-X @ w))   # sigmoid
    return (1 / n) * X.T @ (pred - y)

### Implementacion de lso optimizadores (usando NumPy)
# Gradient Descent (GD)
def step_gd(w, grad, lr, state):
    return w - lr * grad, state

# Stochastic Gradient Descent
def step_sgd(w, grad, lr, state):
    return w - lr * grad, state

# Stochastic Gradient Descent with Momentum
def step_sgdm(w, grad, lr, state):
    beta = 0.9
    v = state.get("v", np.zeros_like(w))
    v = beta * v + lr * grad
    return w - v, {"v": v}

# RMS Prop, mantiene el primedio exponencial del cuadrado de los gradientes
def step_rmsprop(w, grad, lr, state):
    alpha, eps = 0.99, 1e-8
    s = state.get("s", np.zeros_like(w))
    s = alpha * s + (1 - alpha) * grad**2
    return w - lr * grad / (np.sqrt(s) + eps), {"s": s}

# ADAM (Adaptive Moment Estimation)
def step_adam(w, grad, lr, state):
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = state.get("t", 0) + 1
    m = beta1 * state.get("m", np.zeros_like(w)) + (1 - beta1) * grad
    v = beta2 * state.get("v", np.zeros_like(w)) + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)   # corrección de sesgo
    v_hat = v / (1 - beta2**t)
    return w - lr * m_hat / (np.sqrt(v_hat) + eps), {"t": t, "m": m, "v": v}

# Mapeado de optimizadores con funcion step
OPTIMIZERS = {
    "GD":       step_gd,
    "SGD":      step_sgd,
    "SGDM":     step_sgdm,
    "RMSProp":  step_rmsprop,
    "Adam":     step_adam,
}

### Ciclo de ENTRENAMIENTO generico
def train(X_tr, y_tr, X_val, y_val, grad_fn, opt_name, lr=1e-3, epochs=200, batch_size=64):
    np.random.seed(42)
    n, d = X_tr.shape
    w = np.zeros(d) # inicialización de pesos en cero
    state = {} # estado interno del optimizador
    step_fn = OPTIMIZERS[opt_name]
    val_losses = []

    for epoch in range(epochs):
        if opt_name == "GD":
            # GD: un solo paso con todo el dataset
            grad = grad_fn(X_tr, y_tr, w)
            w, state = step_fn(w, grad, lr, state)
        else:
            # Mini-batch
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                batch = idx[start:start + batch_size]
                grad  = grad_fn(X_tr[batch], y_tr[batch], w)
                w, state = step_fn(w, grad, lr, state)

        # Pérdida en validación al final de cada epoca
        val_pred = X_val @ w
        if grad_fn == bce_grad:
            val_pred = 1 / (1 + np.exp(-val_pred))   # sigmoid para clasificacion
            val_loss = -np.mean(y_val * np.log(val_pred + 1e-8) + (1 - y_val) * np.log(1 - val_pred + 1e-8))
        else:
            val_loss = np.mean((val_pred - y_val) ** 2)   # MSE para regresión

        val_losses.append(val_loss)

    return w, val_losses
### ENTRENAMIENTO con los 5 optimizadores
EPOCHS = 200
LR     = 1e-3

results_reg = {}
results_clf = {}

print("\nRegresion Lineal")
for opt in OPTIMIZERS:
    w, vl = train(X_tr_r, y_tr_r, X_val_r, y_val_r, mse_grad, opt, lr=LR, epochs=EPOCHS)
    y_pred = X_val_r @ w
    results_reg[opt] = {
        "w": w, "val_losses": vl,
        "val_mse": mean_squared_error(y_val_r, y_pred),
        "val_r2":  r2_score(y_val_r, y_pred),
    }
    print(f"  {opt:8s} → Val MSE: {results_reg[opt]['val_mse']:.4f}  R²: {results_reg[opt]['val_r2']:.4f}")

print("\nRegresion Logistica")
for opt in OPTIMIZERS:
    w, vl = train(X_tr_c, y_tr_c, X_val_c, y_val_c, bce_grad, opt, lr=LR, epochs=EPOCHS)
    y_prob = 1 / (1 + np.exp(-X_val_c @ w))
    y_pred = (y_prob >= 0.5).astype(int)
    results_clf[opt] = {
        "w": w, "val_losses": vl,
        "val_acc": accuracy_score(y_val_c, y_pred),
        "val_f1":  f1_score(y_val_c, y_pred, average="weighted"),
    }
    print(f"  {opt:8s} → Val Acc: {results_clf[opt]['val_acc']:.4f}  F1: {results_clf[opt]['val_f1']:.4f}")

### Elegir los 2 mejores modelos en la VALIDACION

# En regresion, menor MSE es mejor
# En clasificacion, mayor F1 es mejor

top2_reg = sorted(results_reg, key=lambda x: results_reg[x]["val_mse"])[:2]
top2_clf = sorted(results_clf, key=lambda x: results_clf[x]["val_f1"], reverse=True)[:2]

print(f"\nTop 2 Regresion    (menor MSE): {top2_reg}")
print(f"Top 2 Clasificacion (mayor F1): {top2_clf}")

### Evaluar los 2 mejores en TEST

print("\nTEST Regresion Lineal")
for opt in top2_reg:
    y_pred = X_te_r @ results_reg[opt]["w"]
    print(f"  {opt:8s} → MSE: {mean_squared_error(y_te_r, y_pred):.4f}  "
          f"RMSE: {np.sqrt(mean_squared_error(y_te_r, y_pred)):.4f}  "
          f"R²: {r2_score(y_te_r, y_pred):.4f}")

print("\nTEST Regresion Logistica")
for opt in top2_clf:
    y_prob = 1 / (1 + np.exp(-X_te_c @ results_clf[opt]["w"]))
    y_pred = (y_prob >= 0.5).astype(int)
    print(f"  {opt:8s} → Acc: {accuracy_score(y_te_c, y_pred):.4f}  "
          f"F1: {f1_score(y_te_c, y_pred, average='weighted'):.4f}")

### Graficos de perdida en VALIDACION

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Perdida en VALIDACION por Optimizador", fontsize=14, fontweight="bold")
 
for opt in OPTIMIZERS:
    ax1.plot(results_reg[opt]["val_losses"], label=opt)
ax1.set_title("Regresion Lineal (MSE)")
ax1.set_xlabel("Época"); ax1.set_ylabel("MSE"); ax1.legend(); ax1.grid(True, alpha=0.3)
 
for opt in OPTIMIZERS:
    ax2.plot(results_clf[opt]["val_losses"], label=opt)
ax2.set_title("Regresion Logistica (BCE)")
ax2.set_xlabel("Época"); ax2.set_ylabel("BCE Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150, bbox_inches="tight")
plt.show()