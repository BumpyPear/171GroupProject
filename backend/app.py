# backend/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS

# import your model‑wrapping functions here
# e.g. from model_utils import get_correlation, find_outliers

app = Flask(__name__)
CORS(app)  # allow your React frontend to query this API

@app.route("/api/health")
def health_check():
    return jsonify({"status": "ok"})

@app.route("/api/correlation")
def correlation():
    # you’d call into your existing correlation.py logic
    data = get_correlation()  
    return jsonify(data)


# add more routes for random_forest, scaling, wine_statistics, etc.

if __name__ == "__main__":
    # enables auto‑reload and debug output
    app.run(host="0.0.0.0", port=5000, debug=True)
