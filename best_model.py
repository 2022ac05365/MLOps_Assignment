import joblib # type: ignore
import optuna # type: ignore
from sklearn.datasets import load_iris # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the study
study = optuna.create_study(direction='maximize')

# Train the best model
best_model = RandomForestClassifier(**study.best_params)
best_model.fit(X_train, y_train)

# Save the model
joblib.dump(best_model, 'best_model.pkl')
