from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# Load model and feature names
model = joblib.load("Model/best_model.joblib")
feature_names = joblib.load("Model/feature_names.pkl")  # Or use pickle

@app.route("/")
def home():
    return jsonify({"message": "Habitmic API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Get JSON data
        input_df = pd.DataFrame([data], columns=feature_names)
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
