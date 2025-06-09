# === backend/save_models.py ===
import joblib
import pandas as pd
from polyreg import WinePolynomialRegression
from svm import WineQualitySVR
from random_forest_pipeline import train_and_save_rf
from lightgbm_pipeline import train_and_save_lgbm
import os

# basically, the point of this file is to get all of the models and train an instance of each model
# once they are trained, we save a bundle to a pkl file so we can serve that instance of the model in the Flask app
# that way we dont have to retrain the model every time we need it, because that can take a minute.

# each of these "train and save" functions train up the model, then export to pkl file.



def train_and_save_polyreg(degree=2, model_type="ridge", alpha=1.0, scaler_type="standard"):
    #train and save polyreg model
    wine_poly = WinePolynomialRegression(wine_type="white")
    X, y = wine_poly.load_data()
    wine_poly.preprocess_data(X, y, scaler_type=scaler_type)

    X_train_poly, X_test_poly = wine_poly.create_polynomial_features(degree)

    wine_poly.train_model(X_train_poly, model_type=model_type, alpha=alpha)

    bundle_poly = {
        "scaler":    wine_poly.scaler,
        "poly":      wine_poly.poly_features,
        "model":     wine_poly.model,
        "feature_names": wine_poly.feature_names,
    }

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    joblib.dump(bundle_poly, os.path.join(models_dir, "wine_poly_pipeline.pkl"))
    print("-- Saved wine_poly_pipeline.pkl --")

def train_and_save_svm():
    # train and save SVM
    wine_svm = WineQualitySVR()
    X, y = wine_svm.Dataset_Loader()
    wine_svm.DataPreparer(X, y, test_size=0.2, random_state=42)

    wine_svm.TrainModel(kernel="rbf", C=50, gamma="scale", epsilon=0.1)

    bundle_svm = {
        "scaler": wine_svm.scaler,
        "model":  wine_svm.svr_model, 
        "feature_names": wine_svm.feature_names
    }

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    joblib.dump(bundle_svm, os.path.join(models_dir, "wine_svm_pipeline.pkl"))
    print("-- Saved wine_svm_pipeline.pkl --")

# lgbm


if __name__ == "__main__":
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    train_and_save_polyreg(degree=2, model_type="ridge", alpha=1.0, scaler_type="standard")
    train_and_save_svm()
    train_and_save_rf()
    train_and_save_lgbm(
  n_estimators=1000,
  early_stopping_rounds=20,
  test_size=0.2,
  random_state=42
)