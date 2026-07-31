from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import sys

app = Flask(__name__)

# BULLETPROOF: Get the folder where this script is located
SCRIPT_PATH = os.path.abspath(sys.argv[0] if hasattr(sys, 'argv') and len(sys.argv) > 0 else __file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
if SCRIPT_DIR:
    os.chdir(SCRIPT_DIR)

print("Working directory:", os.getcwd())
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data'}), 400
        features = [float(data.get(f, 0)) for f in FEATURES]
        scaled = scaler.transform(np.array(features).reshape(1, -1))
        pred = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0]
        result = "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected"
        return jsonify({'prediction': result, 'prediction_label': int(pred), 'confidence': f"{max(proba)*100:.2f}%"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
