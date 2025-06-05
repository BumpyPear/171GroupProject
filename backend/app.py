# === backend/app.py ===
import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

# ---- STEP 1: load your pickled pipelines on startup ----

# Adjust paths as needed:
BASE_DIR = os.path.dirname("./")              # points at backend/
ARTIFACT_DIR = os.path.join(BASE_DIR, "models")

# 1) Polynomial‐Regression pipeline
poly_path = os.path.join(ARTIFACT_DIR, "wine_poly_pipeline.pkl")
poly_bundle = joblib.load(poly_path)

poly_scaler       = poly_bundle["scaler"]
poly_transformer  = poly_bundle["poly"]
poly_model        = poly_bundle["model"]
poly_features     = poly_bundle["feature_names"]

# 2) SVM pipeline
svm_path = os.path.join(ARTIFACT_DIR, "wine_svm_pipeline.pkl")
svm_bundle = joblib.load(svm_path)
svm_scaler     = svm_bundle["scaler"]
svm_model      = svm_bundle["model"]
svm_features   = svm_bundle["feature_names"]

# random forests pipeline
rf_path = os.path.join(ARTIFACT_DIR, "wine_rf_pipeline.pkl")
rf_bundle = joblib.load(rf_path)
rf_scaler     = rf_bundle["scaler"]
rf_model      = rf_bundle["model"]
rf_features   = rf_bundle["feature_names"]

# ---- STEP 2: helper function to build a DataFrame row from JSON ----

def json_to_df(json_data: dict, feature_list: list):
    """
    Take a JSON payload like
      { "fixed_acidity": 7.4, "volatile_acidity": 0.70, ... }
    and turn it into a single‐row pandas DataFrame with columns in the order of feature_list.
    """
    missing_keys = [f for f in feature_list if f not in json_data]
    if missing_keys:
        raise KeyError(f"Missing required features: {missing_keys}")

    # Extract values in the correct order:
    row = [json_data[f] for f in feature_list]
    df = pd.DataFrame([row], columns=feature_list)
    return df

# ---- STEP 3: endpoints ----

@app.route("/api/features/polyreg", methods=["GET"])
def get_polyreg_features():
    """
    Return the list of feature‐names that the polynomial‐regression model expects.
    Front end can use this to render input fields dynamically.
    """
    return jsonify({"features": poly_features})


@app.route("/api/predict/polyreg", methods=["POST"])
def predict_polyreg():
    """
    Expects a JSON body that includes exactly all the keys in poly_features.
    E.g.
    {
      "fixed_acidity": 7.4,
      "volatile_acidity": 0.70,
      ...
      "alcohol": 9.4
    }
    Returns:
      { "predicted_quality": 5.7324 }
    """
    try:
        json_data = request.get_json(force=True)
        # Convert JSON → single‐row DataFrame
        X_df = json_to_df(json_data, poly_features)

        # 1) scale
        X_scaled = poly_scaler.transform(X_df)

        # 2) polynomial features
        X_poly   = poly_transformer.transform(X_scaled)

        # 3) predict (regression)
        y_pred = poly_model.predict(X_poly)  # this returns a 1‐element array
        result = float(y_pred[0])  # cast to native Python float for JSON serialization

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


@app.route("/api/features/rf", methods=["GET"])
def get_rf_features():
    return jsonify({"features": rf_features})

@app.route("/api/predict/rf", methods=["POST"])
def predict_rf():
    try:
        json_data = request.get_json(force=True)
        X_df = json_to_df(json_data, rf_features)

        # no scaler so this is commented out:
        # X_scaled = rf_scaler.transform(X_df)
        # y_hat    = rf_model.predict(X_scaled)
        # Otherwise, skip scaling:
        y_hat = rf_model.predict(X_df)

        return jsonify({"predicted_quality": float(y_hat[0])})

    except KeyError as ke:
        return jsonify({"error": "missing_features", "message": str(ke)}), 400
    except ValueError as ve:
        return jsonify({"error": "invalid_input", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "prediction_failed", "message": str(e)}), 500


@app.route("/api/features/svm", methods=["GET"])
def get_svm_features():
    return jsonify({"features": svm_features})


@app.route("/api/predict/svm", methods=["POST"])
def predict_svm():
    """
    Expects a JSON body with same structure (feature‐name: value).
    Returns something like:
      {
         "predicted_class": "High",
         "probability": 0.87
      }
    If the SVM kernel was not fitted with probability=True, you can just return
      { "predicted_class": "High" }.
    """
    try:
        json_data = request.get_json(force=True)
        X_df = json_to_df(json_data, svm_features)

        # 1) scale
        X_scaled = svm_scaler.transform(X_df)

        # 2) predict class
        y_hat = svm_model.predict(X_scaled)   # e.g. array(["High"], dtype=object)
        y_hat_label = y_hat[0]

        # 3) optionally predict probabilities (if the model was trained with .probability=True)
        prob = None
        if hasattr(svm_model, "predict_proba"):
            try:
                # If your SVC was fit with `probability=True`, this will work:
                prob_array = svm_model.predict_proba(X_scaled)  # shape (1, 2)
                # Let’s assume index 0=“High” & index 1=“Low” or vice‐versa; clarify by looking at svm_model.classes_
                classes = list(svm_model.classes_)  # e.g. ["High", "Low"]
                idx = classes.index(y_hat_label)
                prob = float(prob_array[0][idx])
            except:
                prob = None

        response = {"predicted_class": y_hat_label}
        if prob is not None:
            response["probability"] = prob

        return jsonify(response)

    except KeyError as ke:
        return (jsonify({"error": "missing_features", "message": str(ke)}), 400)
    except Exception as e:
        return (jsonify({"error": "prediction_failed", "message": str(e)}), 500)


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "server is up"})


if __name__ == "__main__":
    # “host='0.0.0.0'” makes it visible on your LAN; keep port=5000 unless you have a reason to change.
    app.run(host="0.0.0.0", port=5000, debug=True)
