import os
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def save_model(model, filename):
    """Serialize and store the trained model object."""
    joblib.dump(model, filename)


def load_model(filename):
    """Load a previously saved model from disk."""
    return joblib.load(filename)


def load_and_partition():
    """
    Load California housing data and create a train-test split.
    Returns:
        X_train, X_test, y_train, y_test
    """
    dataset = fetch_california_housing()
    X, y = dataset.data, dataset.target
    return train_test_split(X, y, test_size=0.2, random_state=42)


def init_model():
    """Instantiate a new linear regression estimator."""
    return LinearRegression()


def evaluate_outputs(actual, predicted):
    """
    Evaluate model performance using R² and MSE metrics.
    Returns:
        r2_score, mean_squared_error
    """
    r2_val = r2_score(actual, predicted)
    mse_val = mean_squared_error(actual, predicted)
    return r2_val, mse_val

import numpy as np

def float_to_u8(arr):
    """
    Quantize a float array to uint8 using per-element min-max scaling.
    Returns:
        - quantized uint8 array
        - per-element min values
        - per-element max values
    """
    arr = np.asarray(arr)
    min_vals = arr.copy()
    max_vals = arr.copy()
    quantized = np.zeros_like(arr, dtype=np.uint8)

    for i in range(arr.size):
        val = arr.flat[i]
        lo = min_vals.flat[i]
        hi = max_vals.flat[i]
        if hi == lo:
            quantized.flat[i] = 0
        else:
            scale = 255.0 / (hi - lo)
            quantized.flat[i] = np.round((val - lo) * scale).astype(np.uint8)

    return quantized, min_vals.astype(float), max_vals.astype(float)


def u8_to_float(u8_array, lo_range, hi_range):
    """
    Dequantize uint8 array back to float using per-element min-max ranges.
    Returns:
        - reconstructed float array
    """
    u8_array = np.asarray(u8_array, dtype=np.uint8)
    lo_range = np.asarray(lo_range)
    hi_range = np.asarray(hi_range)
    restored = np.zeros_like(u8_array, dtype=np.float32)

    for i in range(u8_array.size):
        lo = lo_range.flat[i]
        hi = hi_range.flat[i]
        if hi == lo:
            restored.flat[i] = lo
        else:
            scale = (hi - lo) / 255.0
            restored.flat[i] = u8_array.flat[i] * scale + lo

    return restored
