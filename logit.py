"""Logit и Probit модели."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

np.random.seed(42)
n = 500

# Данные: решение о покупке
income = np.random.normal(50, 15, n)
age = np.random.randint(18, 70, n)

# Латентная переменная
latent = 0.05 * income - 0.03 * age + np.random.normal(0, 1, n)
y = (latent > 0).astype(int)

X = sm.add_constant(pd.DataFrame({"income": income, "age": age}))

# === 1. Logit ===
logit_model = sm.Logit(y, X).fit(disp=0)
print("=== Logit ===")
print(logit_model.summary())

# === 2. Probit ===
probit_model = sm.Probit(y, X).fit(disp=0)
print("\n=== Probit ===")
print(probit_model.summary())

# === 3. Сравнение коэффициентов ===
comparison = pd.DataFrame({
    "Logit": logit_model.params,
    "Probit": probit_model.params
})
print("\n=== Сравнение ===")
print(comparison)

# === 4. Marginal effects ===
mfx = logit_model.get_margeff()
print("\n=== Marginal Effects (Logit) ===")
print(mfx.summary())

# === 5. ROC-кривая ===
y_pred = logit_model.predict(X)
fpr, tpr, _ = roc_curve(y, y_pred)
auc = roc_auc_score(y, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC-кривая (Logit)")
plt.legend()
plt.savefig("roc.png")
plt.show()

print(f"\nAUC: {auc:.4f}")
