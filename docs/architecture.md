# Architecture

```mermaid
flowchart LR
    subgraph Source
        UCI["UCI Online Retail II\n(xlsx, downloaded once)"]
    end

    subgraph Landing["Landing (simulated incremental extracts)"]
        SPLIT["scripts/split_landing_files.py"]
        CSV["data/landing/*.csv\n(one file per month)"]
        SPLIT --> CSV
    end

    subgraph Orchestration["Apache Airflow (docker-compose)"]
        DAG["retail_etl DAG\nextract >> validate >> transform"]
    end

    subgraph Warehouse["PostgreSQL: retail_dw"]
        RAW["raw.sales_raw"]
        STG["staging.sales_valid\n+ staging.rejected_records"]
        DW["dw.* star schema\n(dim_customer, dim_product,\ndim_date, dim_geography, fact_sales)"]
        MART["mart.customer_rfm\nmart.sales_monthly"]
        META["meta.ingestion_log"]
    end

    subgraph Analytics
        ST["Streamlit dashboard\n(dashboard/app.py)"]
    end

    subgraph MLOps["MLOps repeat-purchase layer"]
      TRAIN["ml/train_model.py\nfeatures + evaluation"]
      ARTIFACT["artifacts/\nmodel + metrics"]
      API["FastAPI\napi/main.py"]
      SCORE["Batch scoring\nml/score.py"]
      DRIFT["PSI drift report\nml/drift_report.py"]
    end

    UCI --> SPLIT
    CSV --> DAG
    DAG -- extract --> RAW
    DAG -- log every file --> META
    RAW -- validate --> STG
    STG -- transform --> DW
    DW --> MART
    MART --> ST
    DW --> ST
    STG --> ST
    STG --> TRAIN
    TRAIN --> ARTIFACT
    ARTIFACT --> API
    ARTIFACT --> SCORE
    ARTIFACT --> DRIFT
```

## Layers
- **Source**: UCI Online Retail II, downloaded once into `data/raw_download/`.
- **Landing**: `scripts/split_landing_files.py` slices the dataset into 25 monthly CSVs
  under `data/landing/`, standing in for the "simulated monthly extract" incremental source.
- **Ingestion / orchestration**: a single Airflow DAG (`retail_etl`) with three tasks —
  extract, validate, transform — each idempotent and independently re-runnable.
- **Raw layer**: `raw.sales_raw` — unmodified text columns + file provenance.
- **Staging layer**: `staging.sales_valid` (typed, validated) and
  `staging.rejected_records` (failed rows + reason).
- **Warehouse (dw) layer**: star schema — one fact table, four dimensions.
- **Mart layer**: `mart.customer_rfm` and `mart.sales_monthly`, purpose-built for the dashboard.
- **Analytics layer**: Streamlit dashboard reading directly from the mart/dw layers.
- **MLOps layer**: a 90-day repeat-purchase label, temporal customer features, evaluated
  scikit-learn pipeline, versioned Joblib artifact, batch scorer, FastAPI endpoint, and PSI
  feature-drift report.

## Why these choices
- **LocalExecutor / single Postgres container** with two databases (`airflow` for Airflow's
  own metadata, `retail_dw` for the warehouse): the project runs on one machine, so
  CeleryExecutor + Redis + multiple workers would be unused complexity.
- **Streamlit over Tableau Public**: the model API and live warehouse access are easier to
  demonstrate alongside Streamlit than in a static Tableau Public publication.
- **Full dw/mart rebuild each DAG run** rather than incremental dimension merges: staging
  data is small enough (under a million rows) that a full rebuild from `staging.sales_valid`
  is simpler and safer than incremental upsert logic for dimensions. Incremental behavior is
  still real — it happens at the extract/staging layer, where already-loaded files and
  already-staged rows are skipped on re-run.
