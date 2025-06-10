# backend/lightgbm_pipeline.py

import os
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

def train_and_save_lgbm(
    n_estimators: int = 1000,
    early_stopping_rounds: int = 20,
    test_size: float = 0.2,
    random_state: int = 42
):
    repo = fetch_ucirepo(id=186)
    df   = repo.data.original
    dfw  = df[df["color"] == "white"].reset_index(drop=True)
    X    = dfw.drop(columns=["quality", "color"])
    y    = dfw["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        random_state=random_state
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            early_stopping(stopping_rounds=early_stopping_rounds),
            log_evaluation(period=0)      
        ],
    )

    bundle = {
        "model":         model,
        "feature_names": list(X.columns),
    }

    art_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(art_dir, exist_ok=True)
    path = os.path.join(art_dir, "wine_lgbm_pipeline.pkl")
    joblib.dump(bundle, path)
    print(f"-- Saved tuned LightGBM pipeline to {path} --")
