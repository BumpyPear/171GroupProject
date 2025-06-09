import os, joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

# this file uses the same process as save_models.py, but it is specifically for lgbm

def train_and_save_lgbm(n_estimators=100, random_state=42):
    data = fetch_ucirepo(id=186).data.original
    df   = data[data['color']=='white'].reset_index(drop=True)
    X    = df.drop(columns=['quality','color'])
    y    = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)
    
    model = LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    bundle = {
        "model":        model,
        "feature_names": list(X.columns)
    }

    art_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(art_dir, exist_ok=True)
    path = os.path.join(art_dir, "wine_lgbm_pipeline.pkl")
    joblib.dump(bundle, path)
    print(f"-- Saved LightGBM pipeline to {path} --")
