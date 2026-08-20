import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# =========================
# 1. CARREGAR DADOS
# =========================

puree = r'C:\Modelos\MIR_Fruit_purees.csv'
DS = pd.read_csv(puree)

DS_T = DS.set_index("Wavenumbers").T

X = DS_T.values
Y = [col.split('.')[0] for col in DS_T.index]

le = LabelEncoder()
Y_enc = le.fit_transform(Y)

print(DS_T)

# =========================
# 2. NORMALIZAÇÃO
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 3. TREINO / TESTE
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y_enc, test_size=0.2, random_state=42
)

# =========================
# 4. MODELOS
# =========================

modelos = {
    "SVM": SVC(kernel="rbf", C=10, gamma="scale"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# =========================
# 5. TREINAR E COMPARAR
# =========================

resultados = []

for nome, modelo in modelos.items():
    
    # Tempo de treino
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tempo_treino = time.time() - inicio
    
    # Tempo de predição
    inicio = time.time()
    y_pred = modelo.predict(X_test)
    tempo_pred = time.time() - inicio
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n===== {nome} =====")
    print(f"Acurácia: {acc:.4f}")
    print(f"Tempo de treino: {tempo_treino:.4f}s")
    print(f"Tempo de predição: {tempo_pred:.4f}s")
    print("\nRelatório:")
    print(classification_report(y_test, y_pred))
    
    # Salvar resultados
    resultados.append({
        "Modelo": nome,
        "Acurácia": acc,
        "Tempo Treino (s)": tempo_treino,
        "Tempo Predição (s)": tempo_pred
    })

# =========================
# 6. TABELA FINAL
# =========================

df_resultados = pd.DataFrame(resultados)
print("\nResumo Final:")
print(df_resultados.sort_values(by="Acurácia", ascending=False))

# ==============================
# 7. CRIAR DATAFRAME COM SEUS RESULTADOS
# ==============================

df_resultados = pd.DataFrame({
    "Modelo": ["Random Forest", "XGBoost", "KNN", "Decision Tree", "Naive Bayes"],
    "Acurácia": [0.969543, 0.964467, 0.949239, 0.934010, 0.802030],
    "Tempo Treino (s)": [0.835556, 0.531443, 0.004001, 0.130575, 0.004072],
    "Tempo Predição (s)": [0.008998, 0.003001, 1.823732, 0.000000, 0.001377]
})

# ==============================
# 8. GRÁFICO DE ACURÁCIA
# ==============================

plt.figure(figsize=(10, 5))
plt.bar(df_resultados["Modelo"], df_resultados["Acurácia"])

plt.title("Acurácia por Modelo")
plt.xlabel("Modelo")
plt.ylabel("Acurácia")
plt.ylim(0, 1.05)
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)

for i, valor in enumerate(df_resultados["Acurácia"]):
    plt.text(i, valor + 0.01, f"{valor:.3f}", ha="center")

plt.tight_layout()
plt.savefig("grafico_acuracia.png", dpi=300, bbox_inches="tight")
plt.show()

# ==============================
# 9. GRÁFICO DE TEMPO DE TREINO
# ==============================

plt.figure(figsize=(10, 5))
plt.bar(df_resultados["Modelo"], df_resultados["Tempo Treino (s)"])

plt.title("Tempo de Treino por Modelo")
plt.xlabel("Modelo")
plt.ylabel("Tempo de treino (s)")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)

for i, valor in enumerate(df_resultados["Tempo Treino (s)"]):
    plt.text(i, valor + 0.01, f"{valor:.3f}s", ha="center")

plt.tight_layout()
plt.savefig("grafico_tempo_treino.png", dpi=300, bbox_inches="tight")
plt.show()

# ==============================
# 10. GRÁFICO DE TEMPO DE PREDIÇÃO
# ==============================

plt.figure(figsize=(10, 5))
plt.bar(df_resultados["Modelo"], df_resultados["Tempo Predição (s)"])

plt.title("Tempo de Predição por Modelo")
plt.xlabel("Modelo")
plt.ylabel("Tempo de predição (s)")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)

for i, valor in enumerate(df_resultados["Tempo Predição (s)"]):
    plt.text(i, valor + 0.01, f"{valor:.3f}s", ha="center")

plt.tight_layout()
plt.savefig("grafico_tempo_predicao.png", dpi=300, bbox_inches="tight")
plt.show()


# ==============================
# 11. GERANDO A MATRIZ DE CONFUSÃO
# ==============================


melhor_modelo_nome = df_resultados.sort_values(
    by="Acurácia",
    ascending=False
).iloc[0]["Modelo"]

print("Melhor modelo:", melhor_modelo_nome)