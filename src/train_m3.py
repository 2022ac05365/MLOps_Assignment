import mlflow
import mlflow.sklearn
import optuna
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 1000)
    max_depth = trial.suggest_int('max_depth', 1, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 16)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def train_model():
    mlflow.set_experiment("iris_classification")

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    best_value = study.best_value

    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", best_value)

        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        best_model = RandomForestClassifier(**best_params, random_state=42)
        best_model.fit(X_train, y_train)

        mlflow.sklearn.log_model(best_model, "model")
        joblib.dump(best_model, 'models/iris_model.joblib')

        print(f"Best parameters: {best_params}")
        print(f"Best accuracy: {best_value}")


if __name__ == "__main__":
    train_model()
