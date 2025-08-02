
# MLOps Linear Regression Pipeline

This repository implements a complete **MLOps pipeline** for a `LinearRegression` model using the **California Housing Dataset**. The project includes data loading, model training, testing, manual quantization, containerization, and CI/CD integration — all following the guidelines set by Dr. Pratik Mazumder for the MLOps course.

---

## Objective

> Build an MLOps pipeline for Linear Regression including:
> - Data processing  
> - Model training using `LinearRegression`  
> - Manual quantization (8-bit unsigned integer)  
> - Inference from both trained and quantized models  
> - Dockerization and GitHub Actions CI/CD  
> - All work managed on a single branch (`main`)  

---

## Directory Structure
```
|--.github
|     |
|   workflows
|     |
|   ci.yml
|--artifacts
|--src
|   |
|   |--predict.py
|   |--quatize.py
|   |--train.py
|   |--utils.py
|--tests
|   |
|  test_train.py
|--.gitignore
|--Dockerfile
|--Readme.md
|--requirements.txt
```

## Workflow Guide (Step-by-Step)

###  Set up the repository
```bash
git init
touch README.md .gitignore requirements.txt
```

###  Train the model (`src/train.py`)
- Loads California Housing data  
- Trains `LinearRegression`  
- Saves model to `artifacts/model.joblib`

###  Unit testing (`tests/test_train.py`)
- Validates:
  - Data is loaded correctly
  - Model is an instance of `LinearRegression`
  - Model is trained (i.e., has `.coef_`)
  - R² score is above a minimum threshold

###  Manual quantization (`src/quantize.py`)
- Loads model coefficients and bias
- Quantizes weights and bias to `uint8`
- Saves:
  - `artifacts/unquant_params.joblib`
  - `artifacts/quant_params.joblib`
- Performs dequantized inference and prints evaluation

###  Predict (`src/predict.py`)
- Loads:
  - Trained model
  - Quantized parameters
- Performs inference using both versions
- Prints predictions and evaluation metrics

###  Dockerization
Build and run the image:
```bash
docker build -t mlops-app .
docker run --rm mlops-app
```

###  CI/CD with GitHub Actions (`.github/workflows/ci.yml`)
- Runs on every push to `main`
- Steps:
  - Run `pytest`
  - Train model and quantize
  - Build Docker image and run prediction container

---

## Comparison Table – Trained vs Quantized Model

| Metric               | Trained Model | Quantized Model |
|----------------------|---------------|-----------------|
| R² Score             | 0.5758        | 0.5758          |
| Mean Squared Error   | 0.5559        | 0.5559          |
| Max Absolute Error   | 9.8753        | 9.8753          |
| Mean Absolute Error  | 0.5332        | 0.5332          |
| Max Prediction Diff  | —             | 0.000002        |
| Model Size (KB)      | 0.68 KB       | 0.42 KB         |
| Storage Saved (KB)   | —             | 0.26 KB         |
| Max Coeff. Error     | —             | 0.00000002      |
| Bias Error           | —             | 0.00000042      |

---

## Docker Commands

```bash
# Build the Docker image
docker build -t mlops-app .

# Run container and perform prediction
docker run --rm mlops-app
```

---

## Local Development Guide

```bash
# Create environment and install dependencies
python -m venv venv
source venv/bin/activate    
pip install -r requirements.txt
# run the files in the env
python src/train.py
python src/quantize.py
python src/predict.py
```

---


