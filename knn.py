import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    f1_score,
)
import numpy as np

atributos = pd.read_csv("Atributos.csv")

# Atributos extraido
y = atributos["y"].values
ids = atributos["id"].values
atributos_glcm = [c for c in atributos.columns if c.startswith("glcm_")]
atributos_lbp = [c for c in atributos.columns if c.startswith("lbp_")]
atributos_hu = [c for c in atributos.columns if c.startswith("Hu_")]
atributos_forma = [c for c in atributos.columns if c.startswith("forma_")]

# Combinações
atributos_selecao = {
    "GLCM+Forma": atributos_glcm + atributos_forma,
    "LBP+Forma": atributos_lbp + atributos_forma,
    "Momentos_de_Hu+Forma": atributos_hu + atributos_forma,
}

# Configurar K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Machine learn
for nome, atributo in atributos_selecao.items():
    print(f"==========Tipo {nome}==========")
    X = atributos[atributo].values
    acuracia_folds = []
    especificidade_folds = []
    sensibilidade_folds = []
    f1_score_folds = []
    for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ids_test = ids[test_index]
        scaler = StandardScaler()
        X_scaler_train = scaler.fit_transform(X_train)
        X_scaler_test = scaler.transform(X_test)
        # KNN
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(X_scaler_train, y_train)
        y_pred = neigh.predict(X_scaler_test)
        for i in range(len(y_test)):
            real = y_test[i]
            previsto = y_pred[i]
            foto = ids_test[i]
            if real != previsto:
                print(
                    f"Erro: A imagem {foto} é da especie {real}, mas o KNN retornou {previsto}"
                )
        acuracia = accuracy_score(y_test, y_pred)
        sensibilidade = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        matriz_confusão = confusion_matrix(y_test, y_pred)
        especificidade_classes = []
        for i in range(len(matriz_confusão)):
            tp = matriz_confusão[i, i]
            fp = matriz_confusão[:, i].sum() - tp
            fn = matriz_confusão[i, :].sum() - tp
            tn = matriz_confusão.sum() - (tp + fp + fn)
            especificidade_classe = tn / (tn + fp) if (tn + fp) > 0 else 0
            especificidade_classes.append(especificidade_classe)
        especificidade_total = sum(especificidade_classes) / len(especificidade_classes)
        acuracia_folds.append(acuracia)
        especificidade_folds.append(especificidade_total)
        sensibilidade_folds.append(sensibilidade)
        f1_score_folds.append(f1)
    print(f"Acuracia: {np.mean(acuracia_folds):.4%} ± {np.std(acuracia_folds):.4%}")
    print(
        f"Especificidade: {np.mean(especificidade_folds):.4%} ± {np.std(especificidade_folds):.4%}"
    )
    print(
        f"Sensibilidade: {np.mean(sensibilidade_folds):.4%} ± {np.std(sensibilidade_folds):.4%}"
    )
    print(f"F1_score: {np.mean(f1_score_folds):.4%} ± {np.std(f1_score_folds):.4%}")
