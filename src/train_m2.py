import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def train_model(n_estimators, max_depth):
    mlflow.set_experiment("iris_classification")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Load data
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model accuracy: {accuracy}")

        # Save model
        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, 'models/iris_model.joblib')


if __name__ == "__main__":
    train_model(n_estimators=100, max_depth=5)
    train_model(n_estimators=200, max_depth=10)
    train_model(n_estimators=300, max_depth=15)
