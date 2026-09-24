# E-Commerce Sales Analytics and Repeat-Purchase MLOps

Retail ETL pipeline built on the UCI Online Retail II dataset: landing CSVs -> Airflow DAG
-> raw -> staging (validated) -> PostgreSQL star schema -> marts -> Streamlit dashboard.

Part 2 adds a repeat-purchase model with reproducible training, evaluation, batch scoring,
FastAPI inference, and feature-drift reporting.

## Stack
- Python 3.11+ / Pandas for ingestion and transforms
- Apache Airflow (LocalExecutor) for orchestration, via Docker Compose
- PostgreSQL 16 for storage (two databases: `airflow` metadata, `retail_dw` warehouse)
- Streamlit + Plotly for the dashboard
- scikit-learn + Joblib for repeat-purchase prediction
- FastAPI + Uvicorn for model serving

## Prerequisites
- Docker Desktop running
- Python 3.11+ on the host (for the one-time dataset-split script and the dashboard)

## One-time setup

```bash
# 1. Python env (host-side, for split script + dashboard)
python -m venv .venv
source .venv/Scripts/activate   # or .venv/bin/activate on Linux/Mac
pip install -r requirements.txt

# 2. Download the dataset (~45MB) and slice into monthly landing files
mkdir -p data/raw_download
curl -sL -o data/raw_download/online_retail_ii.zip \
  "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
unzip -o data/raw_download/online_retail_ii.zip -d data/raw_download
python scripts/split_landing_files.py   # writes data/landing/2009-12.csv ... 2011-12.csv
```

## Run the pipeline

```bash
# 3. Start Postgres, then Airflow (init once, then scheduler + webserver)
docker compose up -d postgres
docker compose up -d airflow-init        # migrates DB, creates admin/admin user, exits when done
docker compose up -d airflow-scheduler airflow-webserver

# 4. Airflow UI: http://localhost:8080 (user: admin / pass: admin)
#    unpause + trigger "retail_etl", or via CLI:
docker exec -it $(docker compose ps -q airflow-scheduler) airflow dags unpause retail_etl
docker exec -it $(docker compose ps -q airflow-scheduler) airflow dags trigger retail_etl
```

The DAG has three tasks: `extract_landing_files >> validate_and_stage >> transform_to_star_schema`.
All three are idempotent — re-running (or a daily schedule) never duplicates data; already-loaded
files and already-staged rows are skipped automatically.

## Run the dashboard

```bash
source .venv/Scripts/activate
set -a; source .env.local; set +a   # PGHOST=localhost, PGPORT=55432, etc.
streamlit run dashboard/app.py
```

Open http://localhost:8501.

## Train and serve the repeat-purchase model

Run these commands from the repository root after the ETL has populated PostgreSQL:

```bash
set PGHOST=localhost
set PGPORT=55432
set PGDATABASE=retail_dw
set PGUSER=airflow
set PGPASSWORD=airflow
python -m ml.train_model --output-dir artifacts
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The training job creates a 90-day future-purchase label from `staging.sales_valid`, evaluates
on a held-out customer split, and writes `artifacts/repeat_purchase_model.joblib` and
`artifacts/metrics.json`. The API exposes `/health` and `/predict`. Batch scoring is available
with `python -m ml.score <features.csv> --model artifacts/repeat_purchase_model.joblib`.
Use `python -m ml.drift_report <training.csv> <scoring.csv>` to produce a PSI report.

## Dataset source and access

The project uses the free, public UCI Online Retail II dataset:
https://archive.ics.uci.edu/dataset/502/online+retail+ii

The setup script downloads the public archive from UCI, stores the original workbook under
`data/raw_download/`, and creates monthly landing files under `data/landing/`. No final
analytical table is manually edited. The source and transformation rules are documented in
`docs/data_dictionary.md` and `docs/architecture.md`.

## Submission evidence

Run `python -m pytest tests -q` for automated model checks. For execution evidence, capture
the Airflow DAG graph and successful task run at `http://localhost:8080`, the populated mart
tables, the Streamlit dashboard at `http://localhost:8501` (or `8502` if 8501 is occupied),
the printed model metrics, and a successful `/predict` response from the API. A report outline
covering the required 8-10 page submission is in `docs/project_report.md`.

## A note on the Postgres port
`docker-compose.yml` publishes Postgres on host port **55432**, not the default 5432.
During development, something on this machine's network stack (antivirus / security
software, most likely) silently intercepted host-side TCP traffic to port 5432 specifically
— Postgres's own connection log showed zero incoming attempts even while the client reported
"password authentication failed". The exact same credentials worked immediately once the
container was republished on 55432. Container-to-container traffic (Airflow -> Postgres,
both inside the `retail_dw` docker network) is unaffected and still uses the standard 5432
internally. `.env.local` is already set to 55432 for host-side scripts and the dashboard.

## Repo layout
```
docker-compose.yml       # postgres + airflow (init/scheduler/webserver)
db/init/                 # postgres init script (creates retail_dw db + runs schema.sql)
sql/schema.sql           # raw / staging / dw / mart / meta schemas
scripts/
  split_landing_files.py # one-time: xlsx -> monthly landing CSVs
  db.py                  # shared SQLAlchemy engine helper
  extract.py             # landing CSVs -> raw.sales_raw (incremental, logged)
  validate.py             # raw -> staging.sales_valid + staging.rejected_records
  transform.py            # staging -> dw star schema + mart tables
dags/retail_etl_dag.py   # Airflow DAG wiring the three scripts together
dashboard/app.py         # Streamlit dashboard (5 views + data-quality panel)
docs/
  architecture.md        # architecture diagram + design rationale
  data_dictionary.md      # every column, type, source, validation rule
data/
  raw_download/           # downloaded xlsx (gitignored)
  landing/                 # monthly CSVs (gitignored)
```

## Verified end-to-end
- 25 monthly landing files (Dec 2009 - Dec 2011, ~1.07M raw rows) extracted, validated, and
  transformed through the Airflow DAG with all three tasks green.
- 768,873 rows passed validation; 261,822 rejected (missing_customer_id, cancelled_order,
  invalid_quantity, invalid_unit_price) — see `staging.rejected_records` or the dashboard's
  "Data quality" panel.
- Re-running the DAG confirmed idempotent: identical row counts, `extract` reports
  `skipped=25`, `validate` reports nothing new.
- Streamlit dashboard renders all 5 required views (revenue/order trend, top products,
  customer segmentation, country-wise sales, retention/repeat-purchase) against live data.
