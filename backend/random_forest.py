import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# Load and prepare data
wine_quality = fetch_ucirepo(id=186)
df_full = wine_quality.data.original
df_white = df_full[df_full['color'] == 'white'].reset_index(drop=True)
X = df_white.drop(columns=['quality', 'color'])
y = df_white['quality']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def evaluate_tree_count(X_train, y_train, X_test, y_test, max_trees=100, step=5):

    tree_counts = list(range(step, max_trees + 1, step))

    train_mse_history = []
    test_mse_history = []
    train_r2_history = []
    test_r2_history = []

    best_test_r2 = -float('inf')
    optimal_trees = 0
    base_params = {
        'max_depth': 8,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }

    for n_trees in tree_counts:
        rf = RandomForestRegressor(n_estimators=n_trees, **base_params)
        rf.fit(X_train, y_train)

        train_pred = rf.predict(X_train)
        test_pred = rf.predict(X_test)

        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        train_mse_history.append(train_mse)
        test_mse_history.append(test_mse)
        train_r2_history.append(train_r2)
        test_r2_history.append(test_r2)

        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            optimal_trees = n_trees

    return (
        tree_counts,
        train_mse_history,
        test_mse_history,
        train_r2_history,
        test_r2_history,
        optimal_trees,
        best_test_r2
    )

def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [6, 8, 10],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [2, 4, 6],
        'max_features': ['sqrt', 'log2']
    }
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_
best_params = tune_hyperparameters(X_train, y_train)
tuned_model = RandomForestRegressor(
    **best_params,
    random_state=42,
    n_jobs=-1
)

cv_scores = cross_val_score(tuned_model, X_scaled, y, cv=5, scoring='r2')
tree_counts, train_mse_hist, test_mse_hist, train_r2_hist, test_r2_hist, optimal_trees, best_r2 = evaluate_tree_count(
    X_train, y_train, X_test, y_test, max_trees=150, step=5
)

final_model = RandomForestRegressor(
    n_estimators=optimal_trees,
    **{k: v for k, v in best_params.items() if k != 'n_estimators'},
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_train, y_train)

train_pred = final_model.predict(X_train)
test_pred = final_model.predict(X_test)

final_train_mse = mean_squared_error(y_train, train_pred)
final_test_mse = mean_squared_error(y_test, test_pred)
final_train_r2 = r2_score(y_train, train_pred)
final_test_r2 = r2_score(y_test, test_pred)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.plot(tree_counts, train_mse_hist, label='Training MSE', color='blue', linewidth=2)
ax1.plot(tree_counts, test_mse_hist, label='Testing MSE', color='orange', linewidth=2)
ax1.axvline(x=optimal_trees, color='green', linestyle='--', label=f'Optimal Trees ({optimal_trees})')
ax1.set_xlabel('Number of Trees')
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('MSE vs Number of Trees (Wine Quality)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(tree_counts, train_r2_hist, label='Training R²', color='blue', linewidth=2)
ax2.plot(tree_counts, test_r2_hist, label='Testing R²', color='orange', linewidth=2)
ax2.axvline(x=optimal_trees, color='green', linestyle='--', label=f'Optimal Trees ({optimal_trees})')
ax2.set_xlabel('Number of Trees')
ax2.set_ylabel('R² Score')
ax2.set_title('R² vs Number of Trees (Wine Quality)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

feature_importances = final_model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importances)

plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

print(f"Optimal number of trees: {optimal_trees}")
print(f"Final model parameters: {best_params}")
print(f"Training R²: {final_train_r2:.4f}")
print(f"Testing R²: {final_test_r2:.4f}")
print(f"Training MSE: {final_train_mse:.4f}")
print(f"Testing MSE: {final_test_mse:.4f}")
print(f"Gap between Train and Test R²: {final_train_r2 - final_test_r2:.4f}")

print("\nTop 5 most important features:")
top_features = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)[:5]
for feature, importance in top_features:
    print(f"{feature}: {importance:.4f}")
