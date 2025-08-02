import numpy as np
from utils import load_model, load_and_partition, u8_to_float
import joblib
import os
from sklearn.metrics import r2_score, mean_squared_error

def predict_from_actual(model, X_test):
    return model.predict(X_test)

def predict_from_quantized(X_test, quant_file_path):
    params = joblib.load(quant_file_path)

    q_weights = params['quant_weights']
    w_min = params['weights_min']
    w_max = params['weights_max']

    q_bias = params['quant_bias']
    b_min = params['bias_min']
    b_max = params['bias_max']

    weights = u8_to_float(q_weights, w_min, w_max)
    bias = u8_to_float(np.array([q_bias]), np.array([b_min]), np.array([b_max]))[0]

    return X_test @ weights + bias

def print_comparison(y_test, actual_preds, quant_preds):
    print("\n Prediction Comparison (First 5 Samples):")
    for i in range(5):
        print(f"[{i}] Actual: {y_test[i]:.3f} | Trained: {actual_preds[i]:.3f} | Quantized: {quant_preds[i]:.3f}")

    print("\n Evaluation:")
    r2_actual = r2_score(y_test, actual_preds)
    r2_quant = r2_score(y_test, quant_preds)
    mse_actual = mean_squared_error(y_test, actual_preds)
    mse_quant = mean_squared_error(y_test, quant_preds)

    print(f"R² Score (Trained Model): {r2_actual:.4f}")
    print(f"R² Score (Quantized Model): {r2_quant:.4f}")
    print(f"MSE (Trained Model): {mse_actual:.4f}")
    print(f"MSE (Quantized Model): {mse_quant:.4f}")

def main():
    _, X_test, _, y_test = load_and_partition()

    print("Loading trained model...")
    model = load_model("artifacts/model.joblib")
    actual_preds = predict_from_actual(model, X_test)

    quant_path = "artifacts/quant_params.joblib"
    if not os.path.exists(quant_path):
        raise FileNotFoundError("Quantized parameters not found. Run quantize.py first.")

    print("Loading and inferring with quantized parameters...")
    quant_preds = predict_from_quantized(X_test, quant_path)

    print_comparison(y_test, actual_preds, quant_preds)

if __name__ == "__main__":
    main()
