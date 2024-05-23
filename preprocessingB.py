import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Função para carregar o conjunto de dados a partir de um arquivo CSV
def loadDataset(filename, deli=','):
    # Lê o arquivo CSV e armazena os dados em um DataFrame do pandas
    baseDeDados = pd.read_csv(filename, delimiter=deli)
    # Extrai as features (X) e os rótulos (y) do DataFrame
    X = baseDeDados.iloc[:, 1:].values
    y = baseDeDados.iloc[:, 0].values
    return X, y

# Função para dividir o conjunto de dados em conjuntos de treinamento e teste
def splitTrainTestSets(X, y, test_size=0.2):
    # Divide os dados em conjuntos de treinamento e teste
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=test_size)
    return XTrain, XTest, yTrain, yTest

# Função para aplicar a padronização (scaling) aos conjuntos de dados
def computeScaling(train, test):
    # Inicializa o objeto StandardScaler para realizar a padronização
    scaler = StandardScaler()
    # Ajusta (fit) e transforma (transform) os dados de treinamento
    train = scaler.fit_transform(train)
    # Transforma os dados de teste usando os parâmetros aprendidos do conjunto de treinamento
    test = scaler.transform(test)
    return train, test