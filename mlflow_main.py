import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def save_iris_dataset():
    data = load_iris()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df['target'] = data.target
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/iris.csv', index=False)
    print("Iris dataset saved to data/iris.csv")

def load_data_from_csv():
    df = pd.read_csv('data/iris.csv')
    X = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(X, y, test_size=0.2)

def train_and_log_model(params):
    X_train, X_test, y_train, y_test = load_data_from_csv()
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    with open("mlflow_logs.md", "a") as f:
        f.write(f"## Run with params: {params}\n")
        f.write(f"- Accuracy: {accuracy}\n")
        f.write("\n")

    print("Model trained and logged with accuracy:", accuracy)

if __name__ == "__main__":
    save_iris_dataset()
    mlflow.set_experiment("Iris_Classification")
    for params in [{"n_estimators": 100, "max_depth": 3}, {"n_estimators": 200, "max_depth": 5}]:
        with mlflow.start_run():
            train_and_log_model(params)
