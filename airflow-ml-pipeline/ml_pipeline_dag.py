from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path("/tmp/airflow_ml_pipeline")
BASE_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH = BASE_DIR / "raw_data.csv"
TRAIN_DATA_PATH = BASE_DIR / "train_data.csv"
TEST_DATA_PATH = BASE_DIR / "test_data.csv"
MODEL_PATH = BASE_DIR / "logistic_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.txt"


def ingest_data():
    data = load_diabetes(as_frame=True)
    df = data.frame

    df["target"] = (df["target"] > df["target"].median()).astype(int)
    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Data saved to {RAW_DATA_PATH}")


def preprocess_data():
    df = pd.read_csv(RAW_DATA_PATH)

    df = df.dropna()

    X = df.drop(columns=["target"])
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df["target"] = y.values

    train_df, test_df = train_test_split(
        processed_df,
        test_size=0.2,
        random_state=42,
        stratify=processed_df["target"]
    )

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print(f"Training data saved to {TRAIN_DATA_PATH}")
    print(f"Testing data saved to {TEST_DATA_PATH}")


def train_model():
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")


def evaluate_model():
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    with open(METRICS_PATH, "w") as f:
        f.write(f"accuracy={accuracy:.4f}\n")

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Metrics saved to {METRICS_PATH}")


with DAG(
    dag_id="ml_pipeline",
    description="End-to-end ML pipeline with data ingestion, preprocessing, training, and evaluation",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ml", "airflow", "pipeline"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    ingest_task >> preprocess_task >> train_task >> evaluate_task
