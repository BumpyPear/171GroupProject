import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from ucimlrepo import fetch_ucirepo
import numpy as np
import lightgbm as lgb
import pandas as pd

# Fetch the dataset
wine_quality = fetch_ucirepo(id=186)
df_full = wine_quality.data.original
df_white = df_full[df_full['color'] == 'white'].reset_index(drop=True)
X = df_white.drop(columns=['quality', 'color'])
y = df_white['quality']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fix: retain raw data to allow training continuation
train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)

def train_lightgbm_with_tracking(train_data, test_data, X_train, y_train, X_test, y_test, n_epochs=100):
    train_mse_history = []
    test_mse_history = []
    train_r2_history = []
    test_r2_history = []

    model = None
    for epoch in range(1, n_epochs + 1):
        model = lgb.train(
            params={
                'objective': 'regression',
                'metric': 'mse',
                'verbosity': -1,
                'seed': 42
            },
            train_set=train_data,
            valid_sets=[train_data, test_data],
            num_boost_round=1,
            init_model=model,  # warm-start from previous model
            keep_training_booster=True
        )

        # Predictions
        train_pred = model.predict(X_train, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)

        # Metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        train_mse_history.append(train_mse)
        test_mse_history.append(test_mse)
        train_r2_history.append(train_r2)
        test_r2_history.append(test_r2)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, "
                  f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")

    return list(range(1, n_epochs + 1)), train_mse_history, test_mse_history, train_r2_history, test_r2_history

# Train LightGBM
print("Training LightGBM over 100 epochs...")
epochs, train_mse_hist, test_mse_hist, train_r2_hist, test_r2_hist = train_lightgbm_with_tracking(
    train_data, test_data, X_train, y_train, X_test, y_test, n_epochs=100
)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# MSE
ax1.plot(epochs, train_mse_hist, label='Training MSE', color='blue')
ax1.plot(epochs, test_mse_hist, label='Testing MSE', color='orange')
ax1.set_xlabel('Epoch (Number of Boosting Rounds)')
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('MSE Convergence Over Training Epochs (LightGBM)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# R²
ax2.plot(epochs, train_r2_hist, label='Training R²', color='blue')
ax2.plot(epochs, test_r2_hist, label='Testing R²', color='orange')
ax2.set_xlabel('Epoch (Number of Boosting Rounds)')
ax2.set_ylabel('R² Score')
ax2.set_title('R² Convergence Over Training Epochs (LightGBM)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final results
print(f"\nFinal Results after 100 epochs:")
print(f"Training R²: {train_r2_hist[-1]:.4f}")
print(f"Testing R²:  {test_r2_hist[-1]:.4f}")
print(f"Training MSE: {train_mse_hist[-1]:.4f}")
print(f"Testing MSE:  {test_mse_hist[-1]:.4f}")
