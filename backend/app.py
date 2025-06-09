# === backend/app.py ===
import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)


# base paths
BASE_DIR = os.path.dirname("./")             
ARTIFACT_DIR = os.path.join(BASE_DIR, "models")

# this loads all the models and another params from the pkl files

#polyreg pipeline
poly_path = os.path.join(ARTIFACT_DIR, "wine_poly_pipeline.pkl")
poly_bundle = joblib.load(poly_path)

poly_scaler       = poly_bundle["scaler"]
poly_transformer  = poly_bundle["poly"]
poly_model        = poly_bundle["model"]
poly_features     = poly_bundle["feature_names"]

#SVM pipeline
svm_path = os.path.join(ARTIFACT_DIR, "wine_svm_pipeline.pkl")
svm_bundle = joblib.load(svm_path)
svm_scaler     = svm_bundle["scaler"]
svm_model      = svm_bundle["model"]
svm_features   = svm_bundle["feature_names"]

# random forests pipeline
rf_path = os.path.join(ARTIFACT_DIR, "wine_rf_pipeline.pkl")
rf_bundle = joblib.load(rf_path)
rf_model      = rf_bundle["model"]
rf_features   = rf_bundle["feature_names"]

# light gbm pipeline
lgbm_path   = os.path.join(ARTIFACT_DIR, "wine_lgbm_pipeline.pkl")
lgbm_bundle = joblib.load(lgbm_path)
lgbm_model     = lgbm_bundle["model"]
lgbm_features  = lgbm_bundle["feature_names"]


# this converts json data to dataframe
def json_to_df(json_data: dict, feature_list: list):
    """
    Take a JSON payload like
      { "fixed_acidity": 7.4, "volatile_acidity": 0.70, ... }
    and turn it into a single‐row pandas DataFrame with columns in the order of feature_list.
    """
    missing_keys = [f for f in feature_list if f not in json_data]
    if missing_keys:
        raise KeyError(f"Missing required features: {missing_keys}")

    row = [json_data[f] for f in feature_list]
    df = pd.DataFrame([row], columns=feature_list)
    return df


#polyreg endpoints
@app.route("/api/features/polyreg", methods=["GET"])
def get_polyreg_features():
    """
    Return the list of feature‐names that the polynomial‐regression model expects.
    Front end can use this to render input fields dynamically.
    """
    return jsonify({"features": poly_features})


@app.route("/api/predict/polyreg", methods=["POST"])
def predict_polyreg():
    try:
        json_data = request.get_json(force=True)
        X_df = json_to_df(json_data, poly_features)

        X_scaled = poly_scaler.transform(X_df)

        X_poly   = poly_transformer.transform(X_scaled)

        y_pred = poly_model.predict(X_poly)
        result = float(y_pred[0])

        return jsonify({"predicted_quality": result})

    except KeyError as ke:
        return (
            jsonify(
                {
                    "error": "missing_features",
                    "message": str(ke),
                }
            ),
            400,
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "error": "prediction_failed",
                    "message": str(e),
                }
            ),
            500,
        )


# random forests route
@app.route("/api/features/rf", methods=["GET"])
def get_rf_features():
    return jsonify({"features": rf_features})

@app.route("/api/predict/rf", methods=["POST"])
def predict_rf():
    try:
        json_data = request.get_json(force=True)
        X_df = json_to_df(json_data, rf_features)

        y_hat = rf_model.predict(X_df)

        return jsonify({"predicted_quality": float(y_hat[0])})

    except KeyError as ke:
        return jsonify({"error": "missing_features", "message": str(ke)}), 400
    except ValueError as ve:
        return jsonify({"error": "invalid_input", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "prediction_failed", "message": str(e)}), 500


# SVM routes
@app.route("/api/features/svm", methods=["GET"])
def get_svm_features():
    return jsonify({"features": svm_features})


@app.route("/api/predict/svm", methods=["POST"])
def predict_svm():
    try:
        json_data = request.get_json(force=True)
        X_df = json_to_df(json_data, svm_features)
        X_scaled = svm_scaler.transform(X_df)

        y_hat = svm_model.predict(X_scaled) 
        return jsonify({"predicted_quality": float(y_hat[0])})

    except KeyError as ke:
        return (jsonify({"error": "missing_features", "message": str(ke)}), 400)
    except Exception as e:
        return (jsonify({"error": "prediction_failed", "message": str(e)}), 500)
    
#lgbm routes
@app.route("/api/features/lgbm", methods=["GET"])
def get_lgbm_features():
    return jsonify({"features": lgbm_features})

@app.route("/api/predict/lgbm", methods=["POST"])
def predict_lgbm():
    try:
        data = request.get_json(force=True)
        df   = json_to_df(data, lgbm_features)
        # no scaler for trees, so:
        y_hat = lgbm_model.predict(df)
        return jsonify({"predicted_quality": float(y_hat[0])})
    except KeyError as ke:
        return jsonify({"error":"missing_features","message":str(ke)}),400
    except Exception as e:
        return jsonify({"error":"prediction_failed","message":str(e)}),500


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "server is up"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
