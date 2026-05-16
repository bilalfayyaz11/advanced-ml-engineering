# End-to-End Machine Learning Pipeline with Apache Airflow

## Objective
Automate a complete machine learning workflow using Apache Airflow, PostgreSQL, and scikit-learn.

## Project Overview
This project builds an Airflow DAG that orchestrates a full ML pipeline: data ingestion, preprocessing, model training, and model evaluation. Each task runs independently and passes outputs through filesystem artifacts, making the workflow easier to debug and production-friendly.

## Technologies Used
- Python 3.12
- Apache Airflow
- PostgreSQL
- scikit-learn
- pandas
- NumPy
- joblib
- Linux
- Python virtual environments

## Key Skills Demonstrated
- Airflow DAG development
- ML workflow orchestration
- PostgreSQL-backed Airflow setup
- Task dependency management
- Data preprocessing automation
- Model training automation
- Model evaluation tracking
- Pipeline troubleshooting

## Pipeline Flow
```text
ingest_data → preprocess_data → train_model → evaluate_model

