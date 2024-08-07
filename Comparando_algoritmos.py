import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import t

# Load dataset from CSV file
data = pd.read_csv('C:/Users/nicko/PycharmProjects/Comparando/oficialtcld0903.csv')  # Update the path to your CSV file
X = data.drop(columns=['med10'])
y = data['bal75']

def remove_outliers_iqr(data, threshold=1.5):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    data_no_outliers = data[(data >= lower_bound) & (data <= upper_bound)]
    return data_no_outliers

data['bal75'] = remove_outliers_iqr(data['bal75'])
data['bal75'] = data['bal75'].fillna(data['bal75'].median())
y = data['bal75']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "Random Forest Regressor": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Support Vector Regressor": SVR(kernel='linear', C=0.01, epsilon=1.5),  
    "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=10)
}

results_data = []

def smooth_series(series, window_size=15):
    return series.rolling(window=window_size, min_periods=1).mean()

for name, model in models.items():
    print(f"Training and evaluating {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(name)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("\n")
    
    results_data.append({'Model': name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse})


    n = len(y_test)
    se = np.std(y_test - y_pred) / np.sqrt(n)
    t_value = t.ppf(0.975, df=n-1)
    margin_of_error = t_value * se
    lower_bounds = y_pred - margin_of_error
    upper_bounds = y_pred + margin_of_error

    subset_size = 500
    plt.figure(figsize=(10, 6))
    plt.plot(smooth_series(pd.Series(y_test.values[:subset_size])), 'b-', label='Real')
    plt.plot(smooth_series(pd.Series(y_pred[:subset_size])), 'r-', label=f'Predito (R²={model.score(X_test, y_test):.3f}, MAE={mae:.2f})')
    plt.fill_between(range(subset_size), smooth_series(pd.Series(lower_bounds[:subset_size])), smooth_series(pd.Series(upper_bounds[:subset_size])), color='gray', alpha=0.2, label='Intervalo de Predição')
    plt.xlabel('Índice Random Forest')
    plt.ylabel('Valor Random Forest')
    plt.title(f'Valores Reais vs Preditos com Intervalo Conforme ({name})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    try:
        plt.savefig(f'{name}_line_plot_with_interval.png')
        print(f'Gráfico salvo como {name}_line_plot_with_interval.png')
    except Exception as e:
        print(f'Erro ao salvar o gráfico para {name}: {e}')
    
    plt.close()

results_df = pd.DataFrame(results_data)
print("Resultados dos Modelos:")
print(results_df)
