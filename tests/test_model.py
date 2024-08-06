# tests/test_model.py
import joblib
from sklearn.datasets import load_iris


def test_model_prediction():
    model = joblib.load('models/iris_model.joblib')
    X, _ = load_iris(return_X_y=True)
    prediction = model.predict(X[:1])
    assert prediction is not None
