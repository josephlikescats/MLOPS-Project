#!/bin/sh
# Ensure the mlflow database exists, then start the tracking server.
#
# db/init/01_init.sh only runs when the Postgres data volume is created, so a
# project that already had Postgres running before Part 2 would not have an
# mlflow database. Creating it here makes `docker compose up mlflow` work on a
# pre-existing volume as well as a fresh one.
set -e

: "${MLFLOW_DB_HOST:=postgres}"
: "${MLFLOW_DB_PORT:=5432}"
: "${MLFLOW_DB_USER:=airflow}"
: "${MLFLOW_DB_PASSWORD:=airflow}"
: "${MLFLOW_DB_NAME:=mlflow}"
: "${MLFLOW_ARTIFACT_ROOT:=/mlflow/artifacts}"

echo "mlflow: ensuring database ${MLFLOW_DB_NAME} exists on ${MLFLOW_DB_HOST}"
python - <<PY
import sys, time
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

for attempt in range(30):
    try:
        connection = psycopg2.connect(
            host="${MLFLOW_DB_HOST}", port=${MLFLOW_DB_PORT},
            user="${MLFLOW_DB_USER}", password="${MLFLOW_DB_PASSWORD}",
            dbname="postgres",
        )
        break
    except psycopg2.OperationalError as exc:
        print(f"mlflow: waiting for postgres ({attempt + 1}/30): {exc.__class__.__name__}")
        time.sleep(2)
else:
    sys.exit("mlflow: postgres never became reachable")

connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
with connection.cursor() as cursor:
    cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", ("${MLFLOW_DB_NAME}",))
    if cursor.fetchone():
        print("mlflow: database already present")
    else:
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier("${MLFLOW_DB_NAME}")))
        print("mlflow: database created")
connection.close()
PY

mkdir -p "${MLFLOW_ARTIFACT_ROOT}"

exec mlflow server \
    --backend-store-uri "postgresql+psycopg2://${MLFLOW_DB_USER}:${MLFLOW_DB_PASSWORD}@${MLFLOW_DB_HOST}:${MLFLOW_DB_PORT}/${MLFLOW_DB_NAME}" \
    --artifacts-destination "${MLFLOW_ARTIFACT_ROOT}" \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000
