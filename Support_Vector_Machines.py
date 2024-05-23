import preprocessingB as pre
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def convert_IC50_to_common_unit(df):
    if 'IC50' in df.columns:
        df['IC50'] = df['IC50'].apply(lambda x: x / 1000 if x > 100 else x)
    else:
        print("Coluna 'IC50' não encontrada no DataFrame.")

def train_SVM_model_with_kfold_validation(filename):
    X, y = pre.loadDataset(filename)
    
    # Convertendo os valores de IC50 para uma unidade comum (por exemplo, μM)
    df = pd.read_csv(filename)
    convert_IC50_to_common_unit(df)
    X = df.drop(columns=['IC50']).values

    # Hiperparâmetros fornecidos
    best_params = {'C': 344.18492504229994, 'epsilon': 0.06937326899745702, 'gamma': 0.057661861663264986}

    # Avaliação do modelo com K-Fold Cross-Validation usando os melhores parâmetros
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    y_true = []
    y_pred = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, X_test = pre.computeScaling(X_train, X_test)

        svm_model = SVR(C=best_params['C'], gamma=best_params['gamma'], epsilon=best_params['epsilon'])
        svm_model.fit(X_train, y_train)

        y_true.extend(y_test)
        y_pred.extend(svm_model.predict(X_test))

    # Calcular as métricas
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    # Imprimir as métricas
    print("R2 Score:", r2)
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)

    # Plotar gráfico de valores reais vs. previstos
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred, c='blue', label='Valores Previstos')
    plt.plot(y_true, y_true, 'r-', label='Valores Reais')
    plt.title("Valores Reais vs. Previstos de IC50")
    plt.xlabel("Valores Reais")
    plt.ylabel("Valores Previstos")
    plt.legend()
    plt.grid()
    plt.show()

# Chamando a função para treinar o modelo SVM com K-Fold Cross-Validation
train_SVM_model_with_kfold_validation("seu_arquivo")

