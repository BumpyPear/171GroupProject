# === backend/save_models.py ===
import joblib
import pandas as pd
from polyreg import WinePolynomialRegression
from svm import WhiteWineSVM
from random_forest_pipeline import train_and_save_rf
import os

def train_and_save_polyreg(degree=2, model_type="ridge", alpha=1.0, scaler_type="standard"):
    #train and save polyreg model
    wine_poly = WinePolynomialRegression(wine_type="white")
    X, y = wine_poly.load_data()
    X_train, X_test, y_train, y_test = wine_poly.preprocess_data(X, y, scaler_type=scaler_type)

    # Create polynomial features (e.g. degree=2)
    X_train_poly, X_test_poly = wine_poly.create_polynomial_features(degree)

    # Train a ridge regression (for example)
    wine_poly.train_model(X_train_poly, model_type=model_type, alpha=alpha)

    # Now you have:
    # - wine_poly.scaler  ( fitted StandardScaler )  
    # - wine_poly.poly_features  ( fitted PolynomialFeatures )  
    # - wine_poly.model  ( fitted Ridge or LinearRegression )
    #
    # We want to bundle those into one “pipeline object” so that we can call
    #   pipeline.predict(new_features) directly.
    #
    # Option A) Manually bundle into a dict:

    bundle_poly = {
        "scaler":    wine_poly.scaler,
        "poly":      wine_poly.poly_features,
        "model":     wine_poly.model,
        "feature_names": wine_poly.feature_names,  # so we know the order of inputs
    }

    # Save it all into disk:
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    joblib.dump(bundle_poly, os.path.join(models_dir, "wine_poly_pipeline.pkl"))
    print("-- Saved wine_poly_pipeline.pkl --")

def train_and_save_svm():
    # 2) Train and save your SVM pipeline
    wine_svm = WhiteWineSVM()
    X_s, y_s = wine_svm.DataLoader()
    Xs_train, Xs_test, ys_train, ys_test = wine_svm.DataGrouping(X_s, y_s)

    # You can skip hyperparameter tuning here if you already have best params, or run:
    # params, score = wine_svm.hyperparameterTuning(cv=5)
    wine_svm.SVMtraining(kernel="rbf", C=50, gamma="scale")  
    # After this, wine_svm.scaler is fitted, wine_svm.svm_model is fitted.

    bundle_svm = {
        "scaler": wine_svm.scaler,
        "model":  wine_svm.svm_model, 
        "feature_names": wine_svm.feature_names
    }

    # Save it to disk.
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    joblib.dump(bundle_svm, os.path.join(models_dir, "wine_svm_pipeline.pkl"))
    print("-- Saved wine_svm_pipeline.pkl --")

if __name__ == "__main__":
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    train_and_save_polyreg(degree=2, model_type="ridge", alpha=1.0, scaler_type="standard")
    train_and_save_svm()
    train_and_save_rf(n_estimators=100)