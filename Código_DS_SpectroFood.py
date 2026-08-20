import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
# ============================================================
# 1. CARREGAMENTO DO DATASET SPECTROFOOD
# ============================================================

arquivo = r"C:\Modelos\SpectroFood.csv"
df = pd.read_csv(arquivo)
# ============================================================
# 2. LIMPEZA E ORGANIZAÇÃO DO DATA SET
# ============================================================

# Remove linhas completamente vazias
df = df.dropna(how="all")

# Remove duplicatas
df = df.drop_duplicates()
df["Class"] = df["Leek"].str[0]

y = df["Class"]
X = df.drop(columns=["Leek","DRY MATTER","Class"])

# ============================================================
# 3. DEFINIÇÃO DA VARIÁVEL ALVO
# ============================================================


target_column = "Class"

y = df[target_column]

# Mantém apenas as colunas numéricas dos espectros
X = df.drop(columns=[target_column,"DRY MATTER"])
X = X.select_dtypes(include=[np.number])

# ============================================================
# 4. TRATAMENTO DE VALORES AUSENTES
# ============================================================

# Imputação pela média, quando necessária
X = X.fillna(X.mean())

# ============================================================
# 5. CODIFICAÇÃO DAS CLASSES
# ============================================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("Classes:", label_encoder.classes_)

# ============================================================
# 6. DIVISÃO TREINO / TESTE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.25,
    stratify=y_encoded,
    random_state=42
)

print("Treino:", X_train.shape)
print("Teste:", X_test.shape)

# ============================================================
# 7. MODELOS
# ============================================================

models = {

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(n_estimators=300,random_state=42),

    "KNN": KNeighborsClassifier(n_neighbors=5),

    "Naive Bayes": GaussianNB(),

    "XGBoost": XGBClassifier(random_state=42,eval_metric="mlogloss"),

    "SVM": SVC(kernel="rbf",C=10,random_state=42)
}

# ============================================================
# 8. PIPELINE: STANDARDIZATION + PCA + CLASSIFIER
# ============================================================

results = []

trained_models = {}

for model_name, model in models.items():

    pipeline = Pipeline([

        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("classifier", model)

    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    trained_models[model_name] = pipeline

    print("\n=======================================")
    print(model_name)
    print("=======================================")

    print("Accuracy:", accuracy)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )
# ============================================================
# 9. TABELA FINAL DE RESULTADOS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Accuracy",
    ascending=False
)


# Recupera o modelo treinado
rf_model = trained_models["Random Forest"]

# Predições
y_pred_rf = rf_model.predict(X_test)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_rf)

print(cm)

# Visualização
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix - Random Forest")
plt.show()

print(df.head())



cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),
    X,
    y_encoded,
    cv=cv,
    scoring="accuracy"
)

print(scores)
print(scores.mean())