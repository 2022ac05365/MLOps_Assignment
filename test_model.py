import os
from model import train_model


def test_model_training():
    train_model()
    assert os.path.exists('model.joblib')


if __name__ == "__main__":
    test_model_training()
