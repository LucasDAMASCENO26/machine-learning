import preprocessingB as pre # Importa um módulo personalizado chamado preprocessingB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

#Hiperparametros
filename = "arquivo" # Nome do arquivo CSV contendo os dados
numberOfTrees = 10 # Número de árvores na floresta aleatória
md = None # Profundidade máxima das árvores (None significa que não há limite)
mss = 5 # Número mínimo de amostras necessárias para dividir um nó


# Função para treinar um modelo de regressão de floresta aleatória
def computeRandomForestRegressionModel(X, y, numberOfTrees, max_depth, min_samples_split):
    regressor = RandomForestRegressor(random_state=40, n_estimators=numberOfTrees, max_depth=max_depth, min_samples_split=min_samples_split)
    regressor.fit(X, y)
    return regressor


# Função para plotar a importância das características
def plot_feature_importance(regressor, X):
    X, y = pre.loadDataset(filename)                                                    
    regressor = computeRandomForestRegressionModel(X, y, numberOfTrees, max_depth = md, min_samples_split = mss)
    
   
   # Calcula a importância das características usando o modelo treinado
    importances = regressor.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regressor.estimators_], axis=0)
    X_names = pd.read_csv(filename)
    X_names = X_names.drop(['IC50'], axis=1)
    feature_names = X_names.columns.to_list()
    # Criar uma série pandas com as importâncias das features
    forest_importances = pd.Series(importances, index=feature_names, name='Importance')
    # Cria um gráfico de barras mostrando a importância das características
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    sorted_importance = forest_importances.sort_values(ascending=False)

    # Salva a serie pandas   
    sorted_importance.to_csv('sorted_importance2.csv', header=['Importance'])
    # Se os dados de entrada forem uma matriz numpy, converte-os em um DataFrame pandas
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"feature {i}" for i in range(X.shape[1])])
    half_len = len(X.columns) // 2
    first_half = list(range(half_len))
    second_half = list(range(half_len, len(X.columns)))
   #Plotar os gráficos
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))  # 2 subgráficos em uma coluna
    forest_importances.iloc[first_half].plot.bar(yerr=std[first_half], ax=axs[0])
    axs[0].set_title("Feature importances (Parte 1)")
    axs[0].set_ylabel("Mean decrease in impurity")

    forest_importances.iloc[second_half].plot.bar(yerr=std[second_half], ax=axs[1])
    axs[1].set_title("Feature importances (Parte 2)")
    axs[1].set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig('feature_importance_entropy2.png')
    plt.show()

# Função para exibir um gráfico de dispersão entre valores previstos e reais
def showPlot(XPoints, yPoints, XLine, yLine):
    X, y = pre.loadDataset(filename)
    regressor = computeRandomForestRegressionModel(X, y, numberOfTrees, max_depth = md, min_samples_split = mss)
    # Obtendo os valores previstos pelo modelo
    y_predito = regressor.predict(X)
    # Calcula o coeficiente de determinação (R2)
    r2 = r2_score(y, y_predito)

    # Plota um gráfico de dispersão
    plt.scatter(y, y_predito, c='blue', label='Valores Previstos')
    plt.scatter(y, y, c='red', label='Valores Reais')
    plt.title("Regressão de floresta aleatória, R2 = {:.3f}".format(r2))
    plt.xlabel("Descritores")
    plt.ylabel("IC50")
    plt.legend()
    plt.savefig('random_forest_100trees_pIC50_2.png')
    plt.show()

# Função principal para executar a regressão de floresta aleatória
def runRandomForestRegressionExample(filename, numberOfTrees):
    X, y = pre.loadDataset(filename)
    X, y = pre.computeScaling(X, y)
    regressor = computeRandomForestRegressionModel(X, y, numberOfTrees, criterion)

    from sklearn.metrics import r2_score
    r2 = r2_score(y, regressor.predict(X))
    print("Coeficiente de Determinação (R2): {:.3f}".format(r2))

    return regressor, X, y    

if __name__ == "_main_":
    regressor, X, y = runRandomForestRegressionExample(filename, numberOfTrees)
    plot_feature_importance(regressor, X)
    calculatePermutationImportance(regressor, X, y)