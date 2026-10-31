#!/bin/bash
set -e

until pg_isready -U "$POSTGRES_USER" -d postgres >/dev/null 2>&1; do
  echo "waiting for postgres to accept connections..."
  sleep 1
done

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-'EOSQL'
    SELECT 'CREATE DATABASE airflow' WHERE NOT EXISTS (
        SELECT FROM pg_database WHERE datname = 'airflow'
    )\gexec
    SELECT 'CREATE DATABASE retail_dw' WHERE NOT EXISTS (
        SELECT FROM pg_database WHERE datname = 'retail_dw'
    )\gexec
    -- Part 2: MLflow's tracking + model-registry store. The registry is
    -- unavailable on a plain-file backend, so it needs a real database.
    SELECT 'CREATE DATABASE mlflow' WHERE NOT EXISTS (
        SELECT FROM pg_database WHERE datname = 'mlflow'
    )\gexec
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname retail_dw -f /docker-entrypoint-initdb.d/schema.sql
