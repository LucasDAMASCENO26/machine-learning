import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

# Função para converter IC50 para uma unidade comum (por exemplo, μM)
def convert_IC50_to_common_unit(df):
    if 'IC50' in df.columns:
        df['IC50'] = df['IC50'].apply(lambda x: x / 1000 if x > 100 else x)
    else:
        print("Coluna 'IC50' não encontrada no DataFrame.")

# Passo 1: Carregar os dados
filename = "seu_arquivo"
df = pd.read_csv(filename)
convert_IC50_to_common_unit(df)

# Separar as características (features) e o alvo (target)
X = df.drop(columns=['IC50']).values  # Características
y = df['IC50'].values  # Alvo

# Função para aplicar LOOCV e treinar o modelo
def loocv_mlp(hidden_layer_sizes, alpha, learning_rate_init):
    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = MLPRegressor(
            hidden_layer_sizes=(int(hidden_layer_sizes), int(hidden_layer_sizes)),
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            activation='relu',
            solver='adam',
            random_state=1,
            max_iter=1000
        )
        model.fit(X_train_scaled, y_train)
        y_pred.append(model.predict(X_test_scaled)[0])
        y_true.append(y_test[0])

    return r2_score(y_true, y_pred)

# Função de otimização bayesiana
def optimize_mlp():
    def evaluate(hidden_layer_sizes, alpha, learning_rate_init):
        return loocv_mlp(hidden_layer_sizes, alpha, learning_rate_init)
    
    optimizer = BayesianOptimization(
        f=evaluate,
        pbounds={
            'hidden_layer_sizes': (5, 100),
            'alpha': (0.0001, 0.1),
            'learning_rate_init': (0.0001, 0.1)
        },
        random_state=1
    )

    optimizer.maximize(init_points=10, n_iter=90)  # 100 iterações no total

    return optimizer.max['params']

# Encontrar os melhores parâmetros usando otimização bayesiana
best_params = optimize_mlp()
print(f"Melhores parâmetros: {best_params}")

# Avaliar o modelo com LOOCV usando os melhores parâmetros
loo = LeaveOneOut()
# Avaliar o modelo com LOOCV usando os melhores parâmetros
loo = LeaveOneOut()
y_true = []
y_pred = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=(int(best_params['hidden_layer_sizes']), int(best_params['hidden_layer_sizes'])),
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init'],
        activation='relu',
        solver='adam',
        random_state=1,
        max_iter=1000
    )
    model.fit(X_train_scaled, y_train)
    y_pred.append(model.predict(X_test_scaled)[0])
    y_true.append(y_test[0])

# Calcular as métricas
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)

# Imprimir as métricas
print("R^2:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

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
