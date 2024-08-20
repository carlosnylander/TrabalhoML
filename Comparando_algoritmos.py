import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import t

# Load dataset
data = pd.read_csv('C:/Users/nicko/PycharmProjects/Comparando/oficialtcld0903.csv')  # Update the path to your CSV file
X = data.drop(columns=['med10'])
y = data['bal75']

# Dictionary with models
models = {
    "Random Forest Regressor": RandomForestRegressor(n_estimators=50, max_depth=10),
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10),
    "Support Vector Regressor": SVR(kernel='linear', C=0.01, epsilon=1.5),
    "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=10),
    "XGBoost Regressor": XGBRegressor(n_estimators=100, learning_rate=0.1),
    "LightGBM Regressor": LGBMRegressor(n_estimators=100, learning_rate=0.1),
    "Ridge Regression": Ridge(alpha=1.0)
}

results_data = []

def smooth_series(series, window_size=15):
    return series.rolling(window=window_size, min_periods=1).mean()

# Run the process 30 times, but only generate graphs for the last run
for i in range(30):
    print(f"Execution {i+1}/30")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    for name, model in models.items():
        print(f"Training and evaluating {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"{name} - Execution {i+1}")
        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print("\n")
        
        results_data.append({'Model': name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'Run': i+1})

        # Only generate plots during the last run
        if i == 29:
            n = len(y_test)
            se = np.std(y_test - y_pred) / np.sqrt(n)
            t_value = t.ppf(0.975, df=n-1)
            margin_of_error = t_value * se
            lower_bounds = y_pred - margin_of_error
            upper_bounds = y_pred + margin_of_error

            subset_size = 500
            plt.figure(figsize=(12, 6))
            plt.plot(smooth_series(pd.Series(y_test.values[:subset_size])), 'b-', label='Real')
            plt.plot(smooth_series(pd.Series(y_pred[:subset_size])), 'r-', label=f'Predicted (R²={model.score(X_test, y_test):.3f}, MAE={mae:.2f})')

            # Enhancing the prediction interval visibility
            plt.fill_between(range(subset_size),
                             smooth_series(pd.Series(lower_bounds[:subset_size])),
                             smooth_series(pd.Series(upper_bounds[:subset_size])),
                             color='gray', alpha=0.5, label='Prediction Interval')

            plt.xlabel('Index', fontsize=14)
            plt.ylabel('Value', fontsize=14)
            plt.title(f'Real vs Predicted Values with Prediction Interval ({name}) - Execution {i+1}', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', linewidth=0.7)
            plt.tight_layout()

            try:
                plt.savefig(f'{name}_line_plot_with_interval_run_{i+1}_improved.png')
                print(f'Improved plot saved as {name}_line_plot_with_interval_run_{i+1}_improved.png')
            except Exception as e:
                print(f'Error saving the improved plot for {name}: {e}')
            
            plt.close()

# Create a DataFrame with the results
results_df = pd.DataFrame(results_data)

# Generate individual box plots for each algorithm
for model_name in models.keys():
    model_results = results_df[results_df['Model'] == model_name]
    
    plt.figure(figsize=(10, 6))
    model_results.boxplot(column='RMSE', grid=True, patch_artist=True,
                          boxprops=dict(facecolor="lightblue", color="blue"),
                          medianprops=dict(color="red"),
                          whiskerprops=dict(color="black"),
                          capprops=dict(color="black"),
                          flierprops=dict(color="blue", markeredgecolor="blue"))
    
    plt.title(f'RMSE Distribution for {model_name} (30 Runs)', fontsize=18)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.7)
    plt.tight_layout()
    
    try:
        plt.savefig(f'{model_name}_RMSE_Boxplot_30_Runs.png')
        print(f'Boxplot saved as {model_name}_RMSE_Boxplot_30_Runs.png')
    except Exception as e:
        print(f'Error saving boxplot for {model_name}: {e}')
    
    plt.close()

print("Model Results:")
print(results_df)
