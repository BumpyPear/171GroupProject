# === backend/random_forest_pipeline.py ===

import os
import joblib
import pandas as pd
import numpy as np
import sklearn

from sklearn.ensemble import RandomForestRegressor
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

# train and save function for the random forests model

def train_and_save_rf(n_estimators=100, random_state=42):

    wine_quality = fetch_ucirepo(id=186)
    df_full = wine_quality.data.original
    df_white = df_full[df_full['color'] == 'white'].reset_index(drop=True)

    X = df_white.drop(columns=['quality', 'color'])
    y = df_white['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model_input_train = X_train

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(model_input_train, y_train)


    bundle = {
        "model":        rf,
        "feature_names": list(X.columns)
    }

    ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    rf_path = os.path.join(ARTIFACT_DIR, "wine_rf_pipeline.pkl")
    joblib.dump(bundle, rf_path)
    print(f"Saved RandomForest pipeline to {rf_path}")
