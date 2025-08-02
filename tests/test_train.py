import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.train import load_data, train_model
from src.utils import save_model, load_model

MODEL_PATH = "artifacts/model.joblib"
R2_THRESHOLD = 0.5


def test_data_loading():
    X_train, X_test, y_train, y_test = load_data()
    
    assert isinstance(X_train, np.ndarray), "X_train should be a NumPy array"
    assert isinstance(X_test, np.ndarray), "X_test should be a NumPy array"
    assert isinstance(y_train, np.ndarray), "y_train should be a NumPy array"
    assert isinstance(y_test, np.ndarray), "y_test should be a NumPy array"
    
    assert X_train.shape[0] > 10, f"X_train has too few samples: {X_train.shape[0]}"
    assert X_test.shape[0] > 10, f"X_test has too few samples: {X_test.shape[0]}"
    assert y_train.shape[0] == X_train.shape[0], "Mismatch between X_train and y_train sizes"
    assert y_test.shape[0] == X_test.shape[0], "Mismatch between X_test and y_test sizes"
    
    assert X_train.shape[1] == X_test.shape[1], "Feature dimensions should match between train and test sets"

def test_model_instance():
    X_train, _, y_train, _ = load_data()
    model = train_model(X_train, y_train)
    assert isinstance(model, LinearRegression), f"Expected LinearRegression, got {type(model)}"

def test_model_is_trained():
    X_train, _, y_train, _ = load_data()
    model = train_model(X_train, y_train)
    
    assert hasattr(model, "coef_"), "Model missing 'coef_' attribute"
    assert hasattr(model, "intercept_"), "Model missing 'intercept_' attribute"
    
    assert model.coef_.ndim == 1, f"Expected 1D coefficients, got shape: {model.coef_.shape}"
    assert not np.allclose(model.coef_, 0), "Model coefficients are all zero — likely untrained"

def test_model_r2_score():
    _, X_test, _, y_test = load_data()

    if not os.path.exists(MODEL_PATH):
        X_train, _, y_train, _ = load_data()
        model = train_model(X_train, y_train)
        save_model(model, MODEL_PATH)

    model = load_model(MODEL_PATH)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    assert r2 > R2_THRESHOLD, f"R² below threshold: {r2:.4f} < {R2_THRESHOLD}"
    assert r2 <= 1.0, f"R² is unrealistically high: {r2:.4f}"