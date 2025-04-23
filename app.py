from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
from flask_cors import CORS
from dotenv import load_dotenv
import socket

load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# Load model and feature names
model = joblib.load("Model/best_model.joblib")
feature_names = joblib.load("Model/feature_names.pkl")  # Or use pickle

def get_ipv4_address():
    """Get the machine's local IPv4 address (not 127.0.0.1)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

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
    ip_address = get_ipv4_address()
    print(f"🌐 Flask server is running on: http://{ip_address}:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
