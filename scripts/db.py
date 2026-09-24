"""Shared DB connection helper. Reads credentials from environment variables
(set via .env locally, or via docker-compose env_file inside containers)."""
import os
from sqlalchemy import create_engine

def get_engine():
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    db = os.environ.get("PGDATABASE", "retail_dw")
    user = os.environ.get("PGUSER", "airflow")
    pw = os.environ.get("PGPASSWORD", "airflow")
    url = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
    return create_engine(url)
