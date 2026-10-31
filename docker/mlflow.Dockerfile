# MLflow tracking server + model registry, backed by the project's Postgres.
#
# The registry needs a database-backed store, which is why this points at
# Postgres rather than a plain ./mlruns directory. Run artifacts go to a named
# volume so they outlive the container.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# SQLAlchemy 2.1 dropped FallbackAsyncAdaptedQueuePool, which mlflow 2.22.5's
# server-side db store still imports; pin below 2.1 until mlflow catches up.
RUN pip install --no-cache-dir "mlflow>=2.17,<3" "psycopg2-binary>=2.9,<3" "SQLAlchemy>=2.0,<2.1"

COPY docker/mlflow-entrypoint.sh /usr/local/bin/mlflow-entrypoint.sh
RUN chmod +x /usr/local/bin/mlflow-entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["/usr/local/bin/mlflow-entrypoint.sh"]
