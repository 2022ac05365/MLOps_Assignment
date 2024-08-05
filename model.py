import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_model():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.joblib')
    print("Model trained and saved.")


if __name__ == "__main__":
    train_model()
