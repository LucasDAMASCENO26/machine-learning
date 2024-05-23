import preprocessingB as pre
import regressionrandomforest as rf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, explained_variance_score, max_error, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Hiperparâmetros
filename = "arquivo" # Nome do arquivo CSV contendo os dados
numberOfTrees = 10 # Número de árvores na floresta aleatória
md = None # Profundidade máxima das árvores (None significa que não há limite)
mss = 5 # Número mínimo de amostras necessárias para dividir um nó

def runRandomForestRegressionExample(filename):
    # Carregamento do dataset
    X, y = pre.loadDataset(filename, deli=None)
    
    # Nomes dos descritores
    feature_names = ['EState_VSA9', 'MolMR', 'VSA_EState6', 'BCUT2D_CHGLO', 'BertzCT', 'Chi4n', 'PEOE_VSA9', 'BCUT2D_LOGPLOW', 'BalabanJ', 'MinEStateIndex']

    # Construção do modelo Random Forest
    rfModel = rf.computeRandomForestRegressionModel(X, y.ravel(), numberOfTrees, md, mss)
    
    # Visualização do gráfico de valores reais vs. previstos
    rf.showPlot(X, y, X, rfModel.predict(X))
    
    # Previsão
    y_pred = rfModel.predict(X)
    
    # Plot da importância dos descritores
    rf.plot_feature_importance(rfModel, X)

    # Cálculo das métricas de desempenho
    r2 = r2_score(y, y_pred)
    variance_explained = explained_variance_score(y, y_pred)
    max_residual_error = max_error(y, y_pred)
    mean_abs_error = mean_absolute_error(y, y_pred)
    mean_squared_err = mean_squared_error(y, y_pred)

    # Criação de um dicionário com as métricas
    metrics_dict = {
        'Métrica': ['R2', 'Variancia Explicada', 'Erro Residual Máximo', 'Erro Médio Absoluto', 'Erro Quadrático Médio'],
        'Valor': [r2, variance_explained, max_residual_error, mean_abs_error, mean_squared_err]
    }

    # Criação de um DataFrame do Pandas com as métricas
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv('metricas.csv')

    # Impressão da tabela de métricas
    print(metrics_df)

    # Geração do gráfico de importância dos descritores
    feature_importances = rfModel.feature_importances_

    # Ordenar os descritores e as importâncias em ordem decrescente
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    sorted_importances = feature_importances[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_feature_names, sorted_importances)
    plt.xlabel('Descritores')
    plt.ylabel('Importância')
    plt.title('Importância dos Descritores (Ordem Decrescente)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

runRandomForestRegressionExample(filename)