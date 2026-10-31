"""
Churn MLOps DAG: monitor -> decide -> retrain -> score -> re-monitor.

This is the retraining policy expressed as code rather than prose. It runs
weekly and does *not* retrain unconditionally: the monitoring job decides,
using the criteria in ml/config.py (significant feature drift, a performance
metric below its floor, failing input-data quality, or model age), and the
short-circuit skips the retrain when the model is still healthy. Retraining a
healthy model on a schedule is how a working model quietly gets replaced by a
worse one.

    check_model_health -> should_retrain -> retrain_model -> score_customers -> verify_after_retrain

If no model exists yet, check_model_health cannot run its checks and reports
``retrain_recommended = True``, so the first DAG run bootstraps the model.

Requires the ML dependencies inside the Airflow image - see the
_PIP_ADDITIONAL_REQUIREMENTS entry for the scheduler in docker-compose.yml -
and the repo's ml/ and artifacts/ directories mounted under /opt/airflow.
"""
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

# ml/ is mounted at /opt/airflow/ml, so its parent is what goes on the path.
sys.path.insert(0, "/opt/airflow")

DATA_SOURCE = "auto"  # prefer the warehouse, fall back to the landing CSVs

default_args = {
    "owner": "retail_dw",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_model_health(**context):
    """Run every monitoring check and publish the retraining decision."""
    from ml.monitoring import run_monitoring

    try:
        report = run_monitoring(source=DATA_SOURCE)
    except FileNotFoundError as exc:
        # No model artifact yet: this is the bootstrap run.
        print(f"churn_ml: no model to monitor ({exc}) - bootstrapping")
        context["ti"].xcom_push(key="retrain_recommended", value=True)
        context["ti"].xcom_push(key="reasons", value=["no model artifact present"])
        return {"status": "bootstrap", "retrain_recommended": True}

    print(f"churn_ml: monitoring status={report['status']} alerts={report['alerts']}")
    context["ti"].xcom_push(key="retrain_recommended", value=report["retrain_recommended"])
    context["ti"].xcom_push(key="reasons", value=report["alerts"])
    return {
        "status": report["status"],
        "retrain_recommended": report["retrain_recommended"],
        "alerts": report["alerts"],
    }


def should_retrain(**context):
    """Short-circuit the retrain unless monitoring asked for one."""
    ti = context["ti"]
    recommended = ti.xcom_pull(task_ids="check_model_health", key="retrain_recommended")
    reasons = ti.xcom_pull(task_ids="check_model_health", key="reasons") or []
    if recommended:
        print(f"churn_ml: retraining because {reasons}")
    else:
        print("churn_ml: model healthy, skipping retrain")
    return bool(recommended)


def retrain_model(**_):
    """Rebuild features, re-fit every candidate, log to MLflow, register the winner."""
    from ml.train import run_training

    payload = run_training(source=DATA_SOURCE)
    print(
        f"churn_ml: retrained {payload['selected_model']} "
        f"test roc_auc={payload['test']['roc_auc']:.4f} "
        f"version={payload['tracking'].get('model_version')}"
    )
    return {
        "selected_model": payload["selected_model"],
        "model_version": payload["tracking"].get("model_version"),
        "test_roc_auc": payload["test"]["roc_auc"],
    }


def score_customers(**_):
    """Batch-score the current population with the newly registered model."""
    from ml.score_churn import run_scoring

    scored = run_scoring(source=DATA_SOURCE)
    flagged = int(scored["churn_prediction"].sum())
    print(f"churn_ml: scored {len(scored)} customers, {flagged} flagged at risk")
    return {"scored": int(len(scored)), "flagged": flagged}


def verify_after_retrain(**_):
    """Re-run monitoring against the new model so the run records its state."""
    from ml.monitoring import run_monitoring

    report = run_monitoring(source=DATA_SOURCE)
    print(f"churn_ml: post-retrain status={report['status']} alerts={report['alerts']}")
    # Drift against a freshly rebuilt baseline should be near zero. Anything
    # still alerting here is a genuine data problem, not model staleness, so
    # it is surfaced rather than silently accepted.
    if report["alerts"]:
        print("churn_ml: WARNING - alerts persist after retraining, investigate upstream data")
    return {"status": report["status"], "alerts": report["alerts"]}


with DAG(
    dag_id="churn_ml",
    description="Monitor the churn model and retrain it only when a criterion fires",
    default_args=default_args,
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["retail", "part2", "mlops"],
) as dag:

    health_task = PythonOperator(
        task_id="check_model_health",
        python_callable=check_model_health,
    )

    decide_task = ShortCircuitOperator(
        task_id="should_retrain",
        python_callable=should_retrain,
    )

    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )

    score_task = PythonOperator(
        task_id="score_customers",
        python_callable=score_customers,
    )

    verify_task = PythonOperator(
        task_id="verify_after_retrain",
        python_callable=verify_after_retrain,
    )

    health_task >> decide_task >> retrain_task >> score_task >> verify_task
