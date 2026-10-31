# E-Commerce Sales Analytics and Repeat-Purchase MLOps

Retail ETL pipeline built on the UCI Online Retail II dataset: landing CSVs -> Airflow DAG
-> raw -> staging (validated) -> PostgreSQL star schema -> marts -> Streamlit dashboard.

Part 2 adds end-to-end MLOps for **customer-churn prediction**: a 90-day inactivity churn
label on month-end snapshots, three candidate models selected on a chronological split,
MLflow experiment tracking and a model registry, a Dockerized FastAPI service, batch
scoring into the dashboard, and a monitoring job whose retraining criteria drive a weekly
Airflow DAG. Full detail in [docs/mlops.md](docs/mlops.md).

## Stack
- Python 3.11+ / Pandas for ingestion and transforms
- Apache Airflow (LocalExecutor) for orchestration, via Docker Compose
- PostgreSQL 16 for storage (two databases: `airflow` metadata, `retail_dw` warehouse)
- Streamlit + Plotly for the dashboard
- scikit-learn / XGBoost + Joblib for churn and repeat-purchase models
- MLflow for experiment tracking and the model registry
- FastAPI + Uvicorn for model serving, containerized with Docker

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

### Quick start (Windows)

**Double-click `start.bat`**, or drag it into a terminal. It starts Docker if needed, brings
every service up in dependency order, waits for each one to actually be ready, unpauses both
DAGs and opens the dashboard. First run takes 10-15 minutes while the Airflow containers
install their packages; after that it is about a minute.

**Double-click `demo.bat`** (a thin wrapper over `demo.ps1`) for a one-click demo of the whole
project. It does everything `start.bat` does and then some: it is idempotent, so it is safe to
re-run on an already-running stack; it triggers the ETL **only when `dw.fact_sales` is empty**,
rather than re-running it every time; it waits for Postgres, Airflow, MLflow, the API and the
dashboard to each answer HTTP 200 before opening their tab; it opens all four UIs (dashboard,
Airflow, MLflow, API docs) as browser tabs; and it prints a walkthrough cheat sheet with the
URLs, credentials and warehouse row counts. `demo.ps1 -NoBrowser` brings everything up and
reports without opening any tabs or waiting on a keypress, for unattended or CI-style runs.

### Manual start

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
set -a; source .env.local; set +a   # PGHOST=localhost, PGPORT=25432, etc.
streamlit run dashboard/app.py
```

Open http://localhost:8501.

## Part 2: train, serve and monitor the churn model

Run from the repository root. Every command works against PostgreSQL once the ETL has run,
and falls back to the landing CSVs when the warehouse is not up (`--source auto`), so the
model pipeline is runnable without Docker.

```bash
source .venv/Scripts/activate
set -a; source .env.local; set +a       # PGHOST=localhost, PGPORT=25432, etc.

# 1. Train: builds snapshot features, fits logistic regression / random forest / XGBoost,
#    selects on a chronological validation block, tunes the threshold, logs everything to
#    MLflow, registers the winner, and writes artifacts/ + the model card.
python -m ml.train --source auto

# 2. Batch score the current customer population -> artifacts/churn_scores.csv
python -m ml.score_churn --source auto

# 3. Monitor data quality, feature drift, performance, latency and model age
python -m ml.monitoring --source auto

# 4. Serve
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Everything in Docker

```bash
docker compose up -d mlflow                      # MLflow UI + registry: http://localhost:5000
docker compose up -d --build api dashboard       # API :8000, dashboard :8501
```

The `api` and `dashboard` images mount `./artifacts` read-only, so a host-side retrain is
picked up with `curl -X POST http://localhost:8000/reload` — no rebuild needed.

### Try the API

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model                 # version, threshold, features, metrics

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "recency_days": 45, "frequency": 6, "monetary": 1850.40, "average_order_value": 308.40,
  "tenure_days": 520, "distinct_products": 73, "avg_items_per_order": 18.5,
  "avg_interpurchase_days": 62.0, "orders_last_90d": 1, "revenue_last_90d": 210.75,
  "revenue_trend_ratio": 0.42, "distinct_active_months": 5 }'

curl http://localhost:8000/metrics               # request counts, failures, latency p50/p95/p99
curl http://localhost:8000/customers/14889/churn # latest batch score for one customer
```

Interactive docs at http://localhost:8000/docs. The Part 1 repeat-purchase model is still
served at `POST /predict/repeat-purchase`.

### Automated retraining

`dags/churn_ml_dag.py` (`churn_ml`, weekly) runs
`check_model_health -> should_retrain -> retrain_model -> score_customers -> verify_after_retrain`
and **short-circuits the retrain unless a criterion fires** — significant drift (PSI >= 0.25),
precision/recall below 0.55, ROC-AUC below 0.65, failing data quality, or a model older than
30 days. Retraining a healthy model on a schedule is how a working model quietly gets
replaced by a worse one.

```bash
docker exec -it $(docker compose ps -q airflow-scheduler) airflow dags unpause churn_ml
docker exec -it $(docker compose ps -q airflow-scheduler) airflow dags trigger churn_ml
```

For a non-Airflow scheduler, `python -m ml.monitoring --fail-on-alert` exits non-zero when a
criterion fires.

### Churn definition

A customer is **churned** at a cutoff if they place **no order in the following 90 days**.
Only customers who bought at least once in the previous 365 days are scored — someone silent
for longer has already churned rather than being at risk of it. Features are computed strictly
from data on or before the cutoff, and the train/validation/test split cuts the snapshot
cutoffs in time, so no customer's future can leak into training. Configurable in
`ml/config.py`; recorded in every artifact and MLflow run.

### Part 1 repeat-purchase model

Still available and unchanged:

```bash
python -m ml.train_model --output-dir artifacts
python -m ml.score <features.csv> --model artifacts/repeat_purchase_model.joblib
```

## Dataset source and access

The project uses the free, public UCI Online Retail II dataset:
https://archive.ics.uci.edu/dataset/502/online+retail+ii

The setup script downloads the public archive from UCI, stores the original workbook under
`data/raw_download/`, and creates monthly landing files under `data/landing/`. No final
analytical table is manually edited. The source and transformation rules are documented in
`docs/data_dictionary.md` and `docs/architecture.md`.

## Submission evidence

Run `python -m pytest tests -q` for the automated checks (60 tests: churn label and
leakage, the chronological split, PSI and the data-quality rules, the retraining decision,
and the API contract).

For execution evidence, capture the Airflow DAG graphs for `retail_etl` and `churn_ml` at
`http://localhost:8080`, the populated mart tables, the Streamlit dashboard at
`http://localhost:8501` (or `8502` if 8501 is occupied), the MLflow experiment and the
registered model versions at `http://localhost:5000`, the printed training metrics, the
monitoring report, and a successful `/predict` response plus `/metrics` from the API.
A report outline covering the required 8-10 page submission is in
`docs/project_report.md`.

## A note on the Postgres port
`docker-compose.yml` publishes Postgres on host port **25432**, not the default 5432.
Two separate things on this Windows machine made the obvious ports unusable:

- **5432** was silently intercepted by something in the network stack (antivirus / security
  software, most likely) — Postgres's own connection log showed zero incoming attempts even
  while the client reported "password authentication failed".
- **55432**, the first replacement, later fell inside a Windows *reserved* dynamic port range
  (`netsh interface ipv4 show excludedportrange protocol=tcp` listed 55404-55503, claimed by
  Hyper-V/WSL). Docker then failed to bind it outright: "An attempt was made to access a
  socket in a way forbidden by its access permissions."

25432 sits clear of every excluded range. If it is ever taken on another machine, check that
`netsh` output and pick a free port, then update `docker-compose.yml` and `.env.local`
together. Container-to-container traffic (Airflow -> Postgres, both inside the `retail_dw`
docker network) is unaffected and still uses the standard 5432 internally.

## Repo layout
```
docker-compose.yml       # postgres + airflow (init/scheduler/webserver)
.dockerignore            # keeps .venv, .git, mlruns/ and the 45MB workbook out of the build context
db/init/                 # postgres init script (creates retail_dw db + runs schema.sql)
sql/schema.sql           # raw / staging / dw / mart / meta schemas
scripts/
  split_landing_files.py # one-time: xlsx -> monthly landing CSVs
  db.py                  # shared SQLAlchemy engine helper
  extract.py             # landing CSVs -> raw.sales_raw (incremental, logged)
  validate.py             # raw -> staging.sales_valid + staging.rejected_records
  transform.py            # staging -> dw star schema + mart tables
dags/
  retail_etl_dag.py      # Part 1: extract >> validate >> transform
  churn_ml_dag.py        # Part 2: monitor >> decide >> retrain >> score >> verify
ml/
  config.py              # churn window, features, split, thresholds (one place)
  data.py                # transactions from postgres or the landing CSVs
  features.py            # snapshot features + inactivity-window churn label
  models.py              # the 3 candidate pipelines
  metrics.py             # shared metrics + threshold tuning
  train.py               # train -> select -> MLflow -> register -> artifacts
  registry.py            # model resolution for every consumer
  score_churn.py         # batch scoring -> churn_scores.csv
  monitoring.py          # quality / drift / performance / latency + retrain decision
  train_model.py, score.py, drift_report.py   # Part 1 repeat-purchase model
api/
  main.py, Dockerfile    # FastAPI inference service
dashboard/
  app.py, Dockerfile     # Streamlit dashboard (9 views + data-quality panel)
docker/
  mlflow.Dockerfile      # MLflow tracking server + registry, Postgres-backed
docs/
  architecture.md        # architecture diagram + design rationale
  data_dictionary.md     # every column, type, source, validation rule
  mlops.md               # Part 2: lifecycle, registry, monitoring, retraining
data/
  raw_download/          # downloaded xlsx (gitignored)
  landing/               # monthly CSVs (gitignored)
artifacts/               # model, metrics, model card, scores, monitoring report
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

## Part 2 verified end-to-end
- 62,607 labelled customer-month snapshots across 16 month-end cutoffs (May 2010 - Aug 2011)
  built from 768,873 validated transactions; split chronologically 9 / 3 / 4 cutoffs into
  train / validation / test with no cutoff shared between blocks.
- All three candidates trained and logged to MLflow; random forest selected on validation
  ROC-AUC 0.783 and registered as `retail-churn-classifier` v1.
- Held-out test block (Jul-Aug 2011, 17,331 rows): ROC-AUC 0.782, precision 0.677,
  recall 0.952, F1 0.791 at the validation-tuned threshold of 0.34.
- Leakage check: appending a year of future transactions changes no feature value
  (`tests/test_features.py::test_features_use_no_data_after_the_cutoff`).
- 4,269 active customers batch-scored; monitoring reports data quality `pass`, performance
  `pass` (precision 0.599, recall 0.959 on the last closed snapshot), scoring p95 75ms, and
  a correct `significant` drift alert on the time-anchored features over the 3-month gap
  between the last training cutoff and the scoring cutoff.
- 60 automated tests passing.

### Full stack verified in Docker
Every service has now been run end-to-end in Docker from a clean volume:

- `postgres`, `airflow-init`, `airflow-scheduler`, `airflow-webserver`, `mlflow`, `api` and
  `dashboard` all start from `docker compose` and report healthy.
- `retail_etl` ran green from an empty database, producing exactly 1,067,371 raw / 768,873
  valid / 261,822 rejected rows; a second trigger changed no counts (idempotent).
- `churn_ml` ran green through all five tasks — `check_model_health`, `should_retrain`,
  `retrain_model`, `score_customers`, `verify_after_retrain`.
- MLflow's Postgres-backed registry holds `retail-churn-classifier` with versions from both
  the host-side training run and the DAG's retrains.
- The API serves `/health`, `/model`, `/predict`, `/predict/repeat-purchase`, `/metrics`,
  `/reload` and `/customers/{id}/churn` from its container.

Seven bugs were found and fixed in the course of that first clean run and a later rebuild
from deleted images and volumes; they are listed in [docs/mlops.md](docs/mlops.md) and
summarised here:

| Bug | Effect | Fix |
| --- | --- | --- |
| `db/init/01_init.sh` had CRLF endings | `/bin/bash^M: bad interpreter` — the whole DB init silently never ran | converted to LF + `.gitattributes` to stop it recurring |
| SQLAlchemy 2.1 vs mlflow 2.22 | MLflow server crashed on boot (`FallbackAsyncAdaptedQueuePool`) | pinned `SQLAlchemy>=2.0,<2.1` |
| Airflow image ships pandas 2.1.4 | `churn_ml` died on the `"ME"` month-end frequency alias, which only exists from pandas 2.2 | bumping the Airflow pin to `pandas>=2.2,<3` fixed the alias and broke every database call instead (next row); the alias is now resolved version-independently and the pin sits below 2.2 |
| pandas >= 2.2 against Airflow's `SQLAlchemy<2.0` | pandas could not import a SQLAlchemy it accepts, silently treated an `Engine` as a raw DBAPI connection, and every `read_sql`/`to_sql` failed with `'Engine' object has no attribute 'cursor'` — `validate_and_stage` failed and the warehouse stayed empty | `_resolve_frequency()` in `ml/features.py` maps both `"M"` and `"ME"` onto `pd.offsets.MonthEnd()`, valid in every version, so pandas >= 2.2 is no longer needed; both Airflow services repinned to `pandas>=2.1,<2.2` |
| MLflow prints an emoji on Windows | `UnicodeEncodeError` silently skipped **all** experiment logging and model registration | force UTF-8 stdout in `ml/train.py` |
| `retail_etl` allowed concurrent runs | two overlapping runs each passed extract's "already loaded?" check, double-ingesting every file | `max_active_runs=1` |
| `api` and `dashboard` images had no `dill` | `churn_model.joblib` is written inside the Airflow image, which ships dill, so the pickle carries dill references; the slim images failed to load it with `ModuleNotFoundError: No module named 'dill'` and `/health` reported `degraded` with `model_loaded: false` | added `dill>=0.3,<1` to `requirements.txt` and rebuilt both images |
