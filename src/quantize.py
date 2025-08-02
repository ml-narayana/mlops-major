import numpy as np
import joblib
import os
from utils import (
    load_and_partition,
    evaluate_outputs,
    float_to_u8,
    u8_to_float,
    load_model,
)


def quantize_model():
    print(">> Loading the original trained model...")
    model = load_model("artifacts/model.joblib")
    weights = model.coef_
    bias = model.intercept_
    print(f"Original Weights: {weights}")
    print(f"Original Bias: {bias}")

    raw_params = {'weights': weights, 'bias': bias}
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(raw_params, "artifacts/unquant_params.joblib")

    print("\n>> Converting parameters to 8-bit integers...")
    q_weights, w_min, w_max = float_to_u8(weights)
    q_bias, b_min, b_max = float_to_u8(np.array([bias]))

    quant_params = {
        'quant_weights': q_weights,
        'weights_min': w_min,
        'weights_max': w_max,
        'quant_bias': q_bias[0],
        'bias_min': b_min,
        'bias_max': b_max
    }

    joblib.dump(quant_params, "artifacts/quant_params.joblib", compress=3)
    print("Quantized model parameters saved.")

    size_original = os.path.getsize("artifacts/model.joblib")
    size_quantized = os.path.getsize("artifacts/quant_params.joblib")
    print(f"\nModel Size (Original): {size_original / 1024:.2f} KB")
    print(f"Model Size (Quantized): {size_quantized / 1024:.2f} KB")
    dq_weights = u8_to_float(q_weights, w_min, w_max)
    dq_bias = u8_to_float(
        np.array([quant_params['quant_bias']]),
        np.array([quant_params['bias_min']]),
        np.array([quant_params['bias_max']])
    )[0]

    max_weight_error = np.abs(weights - dq_weights).max()
    bias_error = abs(bias - dq_bias)
    print(f"\nCoefficient Error (Max): {max_weight_error:.8f}")
    print(f"Bias Error: {bias_error:.8f}")

    _, X_test, _, y_test = load_and_partition()
    pred_quant = X_test @ dq_weights + dq_bias
    pred_orig = model.predict(X_test)

    print("\n>> Comparing predictions between original and quantized models...")
    diff = np.abs(pred_orig[:5] - pred_quant[:5])
    print(f"Max Prediction Diff: {diff.max():.6f}")
    print(f"Mean Prediction Diff: {diff.mean():.6f}")

    if diff.max() < 0.1:
        print("Good quality in quantization")
    elif diff.max() < 1.0:
        print("Moderate quality in quantization")
    else:
        print("Poor quality in quantization")

    print("\n>> Evaluating quantized model on actual test labels...")
    max_error = np.max(np.abs(y_test - pred_quant))
    mean_error = np.mean(np.abs(y_test - pred_quant))
    r2_score, mse_score = evaluate_outputs(y_test, pred_quant)
    
    print(f"R² Score (Quantized Model): {r2_score:.4f}")
    print(f"MSE (Quantized Model): {mse_score:.4f}")
    print(f"Max Absolute Error: {max_error:.4f}")
    print(f"Mean Absolute Error: {mean_error:.4f}")


if __name__ == "__main__":
    quantize_model()
