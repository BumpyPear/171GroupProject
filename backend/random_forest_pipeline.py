# backend/random_forest_pipeline.py

import os
import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

def train_and_save_rf(
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5
):
    # 1) Load & filter white-wine data
    repo = fetch_ucirepo(id=186)
    df   = repo.data.original
    dfw  = df[df['color']=='white'].reset_index(drop=True)
    X    = dfw.drop(columns=['quality','color'])
    y    = dfw['quality']
    
    # 2) Scale all features
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    # 4) Hyperparameter grid search
    param_grid = {
        'n_estimators':      [50, 100, 150],
        'max_depth':         [6, 8, 10],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf':  [2, 4, 6],
        'max_features':      ['sqrt','log2']
    }
    gs = GridSearchCV(
        RandomForestRegressor(random_state=random_state, n_jobs=-1),
        param_grid=param_grid,
        cv=cv_folds,
        scoring='r2',
        n_jobs=-1
    )
    gs.fit(X_train, y_train)
    best_params = gs.best_params_

    # 5) Evaluate optimal tree count if desired (optional)
    #    You can reuse your evaluate_tree_count function here if you like.
    optimal_trees = best_params['n_estimators']

    # 6) Fit final model with best params
    final_rf = RandomForestRegressor(
        **{k:v for k,v in best_params.items()},
        random_state=random_state,
        n_jobs=-1
    )
    final_rf.fit(X_train, y_train)

    # 7) Bundle scaler, model, and feature names
    bundle = {
        'scaler':        scaler,
        'model':         final_rf,
        'feature_names': list(X.columns)
    }

    # 8) Save to disk
    art_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(art_dir, exist_ok=True)
    path = os.path.join(art_dir, 'wine_rf_pipeline.pkl')
    joblib.dump(bundle, path)
    print(f"-- Saved Random Forest pipeline with tuning to {path} --")
