"""
Retail ETL DAG: extract (landing CSVs -> raw) -> validate (raw -> staging,
with a rejected-records log) -> transform (staging -> dw star schema + marts).

Each task is idempotent: extract skips files already logged as loaded,
validate skips files already staged, transform rebuilds dw/mart from
staging. Re-running the DAG (or backfilling) never duplicates data.
"""
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/scripts")

from extract import extract_new_files
from validate import validate_and_stage
from transform import run_transform

default_args = {
    "owner": "retail_dw",
    "retries": 1,
}

with DAG(
    dag_id="retail_etl",
    description="Landing CSVs -> raw -> staging -> dw star schema -> marts",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["retail", "part1"],
) as dag:

    extract_task = PythonOperator(
        task_id="extract_landing_files",
        python_callable=extract_new_files,
    )

    validate_task = PythonOperator(
        task_id="validate_and_stage",
        python_callable=validate_and_stage,
    )

    transform_task = PythonOperator(
        task_id="transform_to_star_schema",
        python_callable=run_transform,
    )

    extract_task >> validate_task >> transform_task
