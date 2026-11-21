# SCSE3040 MLOps Mini Project

# Fuel Station Daily Sales Prediction

## 1. Project Details

- **Group ID:** G173
- **Problem ID:** MP113
- **Project Title:** Fuel Station Daily Sales (Bhopal)
- **Course:** SCSE3040 - MLOps
- **Faculty:** Richa Sharma
- **Batch:** 29

## 2. Group Members

| Enrollment No. | Name |
|---|---|
| S24CSEU0522 | Shreyas |
| S24CSEU0528 | Shruti Maheshwari |
| S24CSEU0556 | Souradeep Chakraborty |
| S24CSEU0521 | Shreyas Singh |

## 3. Project Objective

The main objective of this project is to develop a machine learning system that predicts the daily fuel sales of fuel stations in Bhopal.

The system predicts the expected quantity of fuel sold per day, measured in litres (`litres_per_day`).

This helps fuel station operators plan their fuel orders, avoid fuel shortages, reduce emergency tanker costs and improve inventory management.

## 4. Problem Statement

Fuel stations need to estimate their daily fuel demand accurately. Incorrect predictions can lead to fuel shortages or unnecessary fuel storage.

This project uses historical fuel station data to build a regression-based machine learning solution that predicts daily fuel sales.

## 5. Dataset Description

This project uses the provided MP113 Fuel Station Daily Sales dataset.

The dataset contains the following files:

| File | Description |
|---|---|
| `train.csv` | Historical data used for model training |
| `later.csv` | Newer data used for monitoring and retraining |
| `holdout_features.csv` | Unseen data used for final predictions |

### Important Features

- `highway_distance_km`
- `nearby_pumps`
- `price_rs`
- `trucks_passing`
- `temp_c`
- `staff`
- `ownership`
- `zone`

**Target Variable:** `litres_per_day`

## 6. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Git and GitHub
- DVC
- MLflow

### Planned MLOps Components

- FastAPI for model serving
- Docker for containerization
- GitHub Actions for CI
- Kubernetes for deployment configuration
- Pytest for testing

## 7. Project Workflow

The project follows an end-to-end MLOps workflow:

1. Data collection and exploration.
2. Data quality analysis and identification of data issues.
3. Data cleaning and preprocessing.
4. Dataset versioning using DVC.
5. Model training and evaluation.
6. Experiment tracking using MLflow.
7. Model registration and version management.
8. Model serving through FastAPI.
9. Containerization using Docker.
10. Automated testing using GitHub Actions.
11. Monitoring, drift detection and model retraining.

## 8. Evaluation Metric

The project uses Mean Absolute Error (MAE) to evaluate model performance.

MAE measures the average difference between actual fuel sales and predicted fuel sales.

A lower MAE indicates better prediction performance.

## 9. Repository Structure

The project is organized into different folders for data, source code, reports, testing and deployment files.

```text
G173_MP113/
│
├── .dvc/
├── .github/workflows/
│   └── ci.yml
├── app/
│   └── main.py
├── data/
│   └── raw/
│       ├── holdout_features.csv
│       ├── later.csv.dvc
│       └── train.csv.dvc
├── models/
│   └── model.joblib
├── reports/
│   ├── findings.yaml
│   └── analysis_output.txt
├── src/
│   ├── check_data.py
│   └── fuel_sales/
│       ├── __init__.py
│       ├── data.py
│       └── train.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .dvcignore
├── .gitignore
├── GROUP.md
├── PROBLEM.md
├── README.md
├── STUDENT_HANDBOOK.md
├── mlflow.db
├── params.yaml
└── requirements.txt
```

Additional files and folders will be added as the MLOps pipeline is developed.

## 10. Project Status

The project is being developed in four stages:

- **Stage 1:** Repository setup, data versioning and data quality analysis.
- **Stage 2:** Data cleaning, model training and MLflow experiment tracking.
- **Stage 3:** API development, Docker, automated testing and Kubernetes configuration.
- **Stage 4:** Model monitoring, drift detection, retraining and final predictions.
