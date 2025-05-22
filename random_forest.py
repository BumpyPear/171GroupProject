import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from ucimlrepo import fetch_ucirepo
import numpy as np

# Fetch the dataset
wine_quality = fetch_ucirepo(id=186)
df_full = wine_quality.data.original
df_white = df_full[df_full['color'] == 'white'].reset_index(drop=True)
X = df_white.drop(columns=['quality', 'color'])
y = df_white['quality']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_with_epoch_tracking(X_train, y_train, X_test, y_test, n_epochs=100):
    """Train Random Forest and track metrics over epochs using warm_start"""
    train_mse_history = []
    test_mse_history = []
    train_r2_history = []
    test_r2_history = []

    # Initialize Random Forest with warm_start=True to enable incremental training
    rf = RandomForestRegressor(
        n_estimators=1,  # Start with 1 estimator
        warm_start=True,  # Enable incremental training
        random_state=42,
        n_jobs=-1  # Use all available cores for faster training
    )

    for epoch in range(1, n_epochs + 1):
        # Set the number of estimators for this epoch
        rf.n_estimators = epoch

        # Fit the model (this will add one more tree due to warm_start)
        rf.fit(X_train, y_train)

        # Make predictions
        train_pred = rf.predict(X_train)
        test_pred = rf.predict(X_test)

        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        # Store metrics
        train_mse_history.append(train_mse)
        test_mse_history.append(test_mse)
        train_r2_history.append(train_r2)
        test_r2_history.append(test_r2)

        # Print progress every 20 epochs
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, "
                  f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")

    return list(range(1, n_epochs + 1)), train_mse_history, test_mse_history, train_r2_history, test_r2_history


# Train the model and get training history over 100 epochs
print("Training Random Forest over 100 epochs...")
epochs, train_mse_hist, test_mse_hist, train_r2_hist, test_r2_hist = train_with_epoch_tracking(
    X_train, y_train, X_test, y_test, n_epochs=100
)

# Create subplots for better visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot MSE over epochs
ax1.plot(epochs, train_mse_hist, label='Training MSE', color='blue', linewidth=2)
ax1.plot(epochs, test_mse_hist, label='Testing MSE', color='orange', linewidth=2)
ax1.set_xlabel('Epoch (Number of Trees)')
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('MSE Convergence Over Training Epochs (Wine Quality)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot R² over epochs
ax2.plot(epochs, train_r2_hist, label='Training R²', color='blue', linewidth=2)
ax2.plot(epochs, test_r2_hist, label='Testing R²', color='orange', linewidth=2)
ax2.set_xlabel('Epoch (Number of Trees)')
ax2.set_ylabel('R² Score')
ax2.set_title('R² Convergence Over Training Epochs (Wine Quality)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final metrics
print(f"\nFinal Results after 100 epochs:")
print(f"Training R²: {train_r2_hist[-1]:.4f}")
print(f"Testing R²:  {test_r2_hist[-1]:.4f}")
print(f"Training MSE: {train_mse_hist[-1]:.4f}")
print(f"Testing MSE:  {test_mse_hist[-1]:.4f}")