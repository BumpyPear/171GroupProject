# === backend/random_forest_pipeline.py ===

import os
import joblib
import pandas as pd
import numpy as np
import sklearn

from sklearn.ensemble import RandomForestRegressor
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

def train_and_save_rf(n_estimators=100, random_state=42):
    """
    This function replaces the “plot‐over‐epochs” code in random_forest.py.
    Instead, it trains one fixed RandomForestRegressor on the same data,
    then pickles the model and feature_names for later inference.
    """
    # 1) load data (only white wine)
    wine_quality = fetch_ucirepo(id=186)
    df_full = wine_quality.data.original
    df_white = df_full[df_full['color'] == 'white'].reset_index(drop=True)

    # 2) split into X, y
    X = df_white.drop(columns=['quality', 'color'])
    y = df_white['quality']

    # 3) train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # 4) (Optional) scale if you want: RandomForest doesn’t strictly need scaling, but if your teammates used it:
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler().fit(X_train)
    # X_train_scaled = scaler.transform(X_train)
    # X_test_scaled  = scaler.transform(X_test)
    # model_input_train = X_train_scaled
    # model_input_test  = X_test_scaled

    # If your existing random_forest.py did NOT scale, you can skip this step:
    model_input_train = X_train
    model_input_test  = X_test

    # 5) instantiate & fit the RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(model_input_train, y_train)

    # 6) Bundle: 
    #    - if you did scale, include scaler in bundle  
    #    - always include rf in bundle  
    #    - include feature_names = list(X.columns) in bundle
    bundle = {
        # "scaler": scaler,                  # uncomment if you used a scaler
        "model":        rf,
        "feature_names": list(X.columns)
    }

    # 7) Save to disk
    ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    rf_path = os.path.join(ARTIFACT_DIR, "wine_rf_pipeline.pkl")
    joblib.dump(bundle, rf_path)
    print(f"Saved RandomForest pipeline to {rf_path}")
