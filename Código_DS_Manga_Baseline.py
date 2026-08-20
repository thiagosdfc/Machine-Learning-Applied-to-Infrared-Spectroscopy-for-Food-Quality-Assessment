import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


arquivo = r"C:\Modelos\Manga2.csv"

df = pd.read_csv(arquivo, sep=";",  decimal=",")
df.columns = df.columns.str.strip()
df.columns = [col.strip() for col in df.columns]


# Alvo
y = df["Cultivo"]

# Features químicas
X = df[[
    "Vit C (mg/100g)",
    "TA (mg/100g)",
    "SSC (oBrix)"]]

# Codificar classes
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Classes:")
print(dict(zip(le.classes_, le.transform(le.classes_))))

# Treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded
)

# Modelos
modelos = {
    "SVM": SVC(kernel="rbf", C=10, gamma="scale"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="mlogloss",
        use_label_encoder=False
    )
    
}

for nome, modelo in modelos.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", modelo)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"\n===== {nome} =====")
    print("Acurácia:", round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
colunas_remover = [
    "No",
    "Cultivo",
    "Vit C (mg/100g)",
    "TA (mg/100g)",
    "SSC (oBrix)",
    "label"
]

X_nir = df.drop(columns=colunas_remover, errors="ignore")

# Garantir que só fiquem colunas numéricas
X_nir = X_nir.select_dtypes(include=[np.number])

y_nir = df["Cultivo"]

print("Formato X_nir:", X_nir.shape)
print("Formato y_nir:", y_nir.shape)

# ==============================
# CLASSIFICAÇÃO COM NIR COMPLETO
# ==============================

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

le_nir = LabelEncoder()
y_nir_encoded = le_nir.fit_transform(y_nir)

X_train_nir, X_test_nir, y_train_nir, y_test_nir = train_test_split(
    X_nir,
    y_nir_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_nir_encoded
)

modelos_nir = {
    "SVM": SVC(kernel="rbf", C=10, gamma="scale"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    
    "Naive Bayes": GaussianNB(),
    
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="mlogloss",
        use_label_encoder=False
    )
}

for nome, modelo in modelos_nir.items():
    pipe_nir = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("modelo", modelo)
    ])

    pipe_nir.fit(X_train_nir, y_train_nir)
    y_pred_nir = pipe_nir.predict(X_test_nir)

    print(f"\n===== NIR COMPLETO - {nome} =====")
    print("Acurácia:", round(accuracy_score(y_test_nir, y_pred_nir), 4))
    print(classification_report(
        y_test_nir,
        y_pred_nir,
        target_names=le_nir.classes_
    ))
    
# ==============================
# BASELINE
# ==============================

resultados_baseline = {}

for nome, modelo in modelos.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", modelo)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    resultados_baseline[nome] = acc


# ==============================
# NIR
# ==============================

resultados_nir = {}

for nome, modelo in modelos_nir.items():
    pipe_nir = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("modelo", modelo)
    ])
    
    pipe_nir.fit(X_train_nir, y_train_nir)
    y_pred_nir = pipe_nir.predict(X_test_nir)
    
    acc_nir = accuracy_score(y_test_nir, y_pred_nir)
    resultados_nir[nome] = acc_nir
    
df_compare = pd.DataFrame({
    "Modelo": list(resultados_baseline.keys()),
    "Baseline (Químico)": list(resultados_baseline.values()),
    "NIR Completo": [resultados_nir[m] for m in resultados_baseline.keys()]
})

display(df_compare)

df_compare.set_index("Modelo").plot(
    kind="bar",
    figsize=(10,6)
)

plt.title("Comparação de Desempenho: Dados Químicos vs Espectroscopia NIR")
plt.ylabel("Acurácia")
plt.xticks(rotation=0)
plt.grid(axis="y")

plt.legend()
plt.tight_layout()
plt.show()

ax = df_compare.set_index("Modelo").plot(kind="bar", figsize=(10,6))

for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.2f}",
        (p.get_x() + p.get_width()/2, p.get_height()),
        ha="center",
        va="bottom"
    )

plt.title("Comparação: Baseline vs NIR")
plt.ylabel("Acurácia")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.show()