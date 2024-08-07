from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

MODEL_PATH = 'models/iris_model.joblib'

# Load the model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Flower names mapping
flower_names = {
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
}

# Feature explanations
feature_explanations = [
    "Sepal length in cm",
    "Sepal width in cm",
    "Petal length in cm",
    "Petal width in cm"
]


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model loading failed'}), 500

    data = request.get_json(force=True)
    features = data['features']
    prediction = model.predict([features])[0]
    flower_name = flower_names[int(prediction)]
    response = {
        'prediction': int(prediction),
        'flower_name': flower_name,
        'features': [
            {'value': features[0], 'description': feature_explanations[0]},
            {'value': features[1], 'description': feature_explanations[1]},
            {'value': features[2], 'description': feature_explanations[2]},
            {'value': features[3], 'description': feature_explanations[3]},
        ]
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
