from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Iris Prediction API. Send a POST request to /predict with features to get a prediction."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'features' not in data:
        return jsonify({'error': 'Please provide features in the request body'}), 400
    try:
        model = joblib.load('models/iris_model_tuned.joblib')
    except Exception as e:
        app.logger.error(f"Failed to load model: {str(e)}")
        return jsonify({'error': 'Model loading failed'}), 500
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)