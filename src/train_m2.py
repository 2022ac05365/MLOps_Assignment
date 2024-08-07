import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

def add_noise(X, y, noise_level=0.1):
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    return X_noisy, y

def train_model(n_estimators, max_depth):
    mlflow.set_experiment("iris_classification")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Load data and add noise
        X, y = load_iris(return_X_y=True)
        X, y = add_noise(X, y)

        # Use stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train model
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        # Calculate mean accuracy
        mean_accuracy = np.mean(accuracies)

        # Log metrics
        mlflow.log_metric("mean_accuracy", mean_accuracy)
        print(f"Model mean accuracy: {mean_accuracy:.4f}")

        # Save model (using the last fold's model)
        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, 'models/iris_model.joblib')

        # Log feature importances
        feature_importance = model.feature_importances_
        for i, importance in enumerate(feature_importance):
            mlflow.log_metric(f"feature_importance_{i}", importance)

if __name__ == "__main__":
    # Try a range of hyperparameters
    train_model(n_estimators=10, max_depth=3)
    train_model(n_estimators=100, max_depth=5)
    train_model(n_estimators=1000, max_depth=10)