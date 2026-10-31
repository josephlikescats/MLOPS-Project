# Part 2: MLOps for customer-churn prediction

This document covers the model lifecycle: how churn is defined, how the model
is trained and selected, how it is tracked and versioned, how it is served,
what is monitored, and the criteria that trigger a retrain.

Part 1 (the ETL pipeline, warehouse and dashboard) is documented in
[architecture.md](architecture.md) and [data_dictionary.md](data_dictionary.md).

---

## 1. Churn definition

> A customer is **churned** at a cutoff date if they place **no order in the
> 90 days that follow it**.

The window is the `CHURN_INACTIVITY_DAYS` setting in `ml/config.py` and is
recorded in every model artifact and MLflow run, so a saved model always
states the label it was trained against.

Three choices make this definition workable rather than merely plausible:

| Choice | Value | Why |
|---|---|---|
| Inactivity window | 90 days | Long enough that a normal gap between orders is not mistaken for churn (the median gap in this dataset is well under 90 days), short enough that a flagged customer is still worth contacting. |
| Eligible population | bought at least once in the previous 365 days | A customer silent for two years has already churned. Including them would let the model score the definition instead of behaviour. |
| Label source | the transaction table only | No survey, no cancellation event, nothing subjective. The label is reproducible from `staging.sales_valid` alone. |

### Snapshots

Churn only means something relative to a point in time, so features are built
at **month-end snapshots** rather than once over the whole history. At each
cutoff, every eligible customer contributes one row:

```
   history (features only)        inactivity window (label only)
   |---------------------------|------------------------------|
                            cutoff                     cutoff + 90d
```

A cutoff is only usable for training while the dataset still covers the whole
window after it, so the final 90 days of data can be **scored but never
trained on**. On the UCI Online Retail II data (Dec 2009 – Dec 2011) this
yields **16 labellable cutoffs** and **62,607 customer-month rows** from
5,245 distinct customers, with an overall churn rate of 57.7%.

---

## 2. Features

Twelve features, all computed strictly from transactions on or before the
cutoff. The order is fixed in `ml/config.py:FEATURES` and stored in the model
artifact, so the API cannot feed the model columns in the wrong order.

| Feature | Definition |
|---|---|
| `recency_days` | days from the customer's last order to the cutoff |
| `frequency` | distinct invoices in history |
| `monetary` | total revenue in history |
| `average_order_value` | `monetary / frequency` |
| `tenure_days` | days from the customer's first order to the cutoff |
| `distinct_products` | distinct `stock_code`s ever bought |
| `avg_items_per_order` | total quantity / `frequency` |
| `avg_interpurchase_days` | mean gap between orders; **null** for single-order customers, median-imputed inside the pipeline |
| `orders_last_90d` | distinct invoices in the 90 days before the cutoff |
| `revenue_last_90d` | revenue in the 90 days before the cutoff |
| `revenue_trend_ratio` | `(revenue_last_90d + 1) / (revenue_prior_90d + 1)` — above 1 is accelerating spend, below 1 decelerating. The `+1` keeps it finite for customers with no spend in the prior window. |
| `distinct_active_months` | distinct calendar months with at least one order |

Imputation and scaling live **inside** the scikit-learn pipeline, not in a
separate preprocessing step. The saved artifact is the whole
transform-and-predict chain, so inference applies exactly the same median fill
the training data produced, and there is no preprocessing step to forget.

---

## 3. Train / validation / test split

The split is **chronological by snapshot cutoff**, not random:

| Block | Cutoffs | Purpose |
|---|---|---|
| Train | earliest 60% | fit the candidates |
| Validation | next 20% | select the model **and** tune the decision threshold |
| Test | latest 20% | reported metrics, read once |

A whole cutoff month lands in exactly one block, so the test block is strictly
later in time than the training block and no customer's future can leak
backwards. `tests/test_features.py::test_features_use_no_data_after_the_cutoff`
asserts this directly: appending a year of future transactions must not change
a single feature value.

Selection and threshold tuning both happen on validation because both are
decisions fitted to data. The test block influences no choice.

---

## 4. Candidates and selection

Three candidates, each a full pipeline (`ml/models.py`):

| Candidate | Configuration |
|---|---|
| `logistic_regression` | median impute → standard scale → `class_weight="balanced"` |
| `random_forest` | 300 trees, `max_depth=12`, `min_samples_leaf=20`, balanced |
| `xgboost` | 400 rounds, `max_depth=5`, `lr=0.05`, `scale_pos_weight` from the training class ratio |

XGBoost is used when installed and sklearn's `HistGradientBoostingClassifier`
substituted when it is not, so the pipeline never fails over an optional
dependency. All three are trained on every run; the winner is the highest
validation ROC-AUC (`SELECTION_METRIC`).

### Threshold

The decision threshold is tuned on the validation block to maximise F1, and
stored in the artifact — the API reads it rather than assuming 0.5.

Note what F1-optimal means on this problem: churn is the **majority** class
(~58% at the test cutoffs), so the F1-optimal threshold is permissive and
favours recall heavily (recall ≈ 0.95, precision ≈ 0.68). That is the right
default for *ranking* a retention list, but it flags a large share of the
base. Before a campaign with a real per-contact cost, retrain with a precision
floor:

```bash
python -m ml.train --min-precision 0.75
```

This constrains threshold tuning to thresholds meeting that precision, trading
recall for a shorter, higher-confidence list. The dashboard's retention view
also lets the threshold be raised interactively without retraining.

---

## 5. Experiment tracking and the model registry

Every training run logs to MLflow: one parent run per job and one nested run
per candidate.

| Logged | Contents |
|---|---|
| Params | every classifier hyperparameter, the churn config, the decision threshold |
| Metrics | validation and test accuracy, precision, recall, F1, ROC-AUC, average precision, Brier score |
| Artifacts | the serialized pipeline with signature and input example, feature importances, the churn config, the split summary |
| Tags | model family, selected flag, data source |

The winner is then registered as a new version of **`retail-churn-classifier`**.

Default tracking store is `sqlite:///mlflow.db` in the repo root — chosen
because the model registry is **unavailable on the plain-file (`./mlruns`)
backend**, and sqlite keeps `register_model` working with no server running.
For the UI and a shared store, run the server and point the URI at it:

```bash
docker compose up -d mlflow                      # http://localhost:5000, Postgres-backed
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Model resolution

`ml/registry.py` is the single resolver every consumer uses — the API, the
batch scorer, the monitoring job. Order:

1. `CHURN_MODEL_URI` — an explicit MLflow URI
   (`models:/retail-churn-classifier/3`, `models:/<name>@champion`,
   `runs:/<run_id>/model`). This is how a deployment pins a version.
2. `CHURN_MODEL_PATH`, else `artifacts/churn_model.joblib` — the local bundle,
   which carries the model, feature order, threshold, metrics and the MLflow
   run and version it came from.
3. the latest registry version, if one is reachable.

The local artifact is preferred over the registry by default so the container
starts with no network dependency while still reporting the registry version
it corresponds to. Resolution never guesses a threshold or a feature order:
both come from the bundle, and a registry-only load reads the threshold back
from the run that produced it.

---

## 6. Serving

`api/main.py`, containerized by `api/Dockerfile`:

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness; reports `degraded` rather than crashing when no model loaded |
| `GET /ready` | readiness; 503 until a model is loaded |
| `GET /model` | version, family, threshold, feature list, churn config, metrics |
| `POST /predict` | churn risk for one customer's features |
| `POST /predict/batch` | many at once, capped by `MAX_BATCH_SIZE` |
| `GET /customers/{id}/churn` | the latest **batch** score for a customer |
| `POST /predict/repeat-purchase` | the Part 1 repeat-purchase model, still available |
| `POST /reload` | re-resolve the model after a retrain, without a restart |
| `GET /metrics` | request counts, failure counts, latency percentiles |
| `GET /metrics/prometheus` | the same in Prometheus text format |

Two deliberate design points:

- **`/predict` computes nothing from the warehouse.** The caller supplies
  features. Building the 12 snapshot features costs a full pass over the
  transaction history, so that work belongs to the scheduled batch job.
  `/customers/{id}/churn` serves its output instead — the usual split of
  expensive batch features and cheap online lookup.
- **Request validation mirrors the data-quality rules.** Negative recency,
  zero frequency and negative revenue are rejected with a 422 at the edge,
  which is the same domain check the monitoring job applies to a batch.

The model is resolved once at startup. A load failure is reported through
`/health` and `/metrics` rather than crash-looping, so the cause is visible.

---

## 7. Monitoring

`python -m ml.monitoring` runs four independent checks, because a churn model
can fail in four ways and only one of them shows up in the model's own metrics.

| Check | What it measures | Alert threshold |
|---|---|---|
| **Input data quality** | null rates, non-finite values, domain violations (negative recency, zero frequency), duplicate customers | any domain violation, or null rate > 5% on a feature that should not be null |
| **Feature drift** | per-feature PSI of the live scoring population against the training baseline | PSI ≥ 0.25 significant (0.10–0.25 moderate) |
| **Model performance** | precision, recall, ROC-AUC on the most recent snapshot whose window has closed | precision < 0.55, recall < 0.55, or ROC-AUC < 0.65 |
| **Scoring latency** | p95 wall-clock of a batch score | p95 > 250 ms |

Thresholds live in `ml/config.py:MonitoringThresholds`.

### Reading the drift numbers

Drift is measured **like-for-like: one cutoff against one cutoff.** The
baseline is the *last training cutoff*, not the pooled training table.
Comparing a single live snapshot against all 16 pooled cutoffs reports huge
drift on every calendar-anchored feature and swamps the real signal — an
earlier version of this job reported PSI 0.94 on `tenure_days` purely as an
artifact of that pooling.

Even like-for-like, expect the time-anchored features to drift as the gap
between training and scoring grows. On the current artifact (trained to
Aug 2011, scoring at Dec 2011):

```
tenure_days             0.53  significant   <- +100 days of tenure for every surviving customer
recency_days            0.32  significant
revenue_trend_ratio     0.30  significant
revenue_last_90d        0.22  moderate
orders_last_90d         0.10  moderate
frequency               0.04  stable
monetary                0.03  stable
average_order_value     0.00  stable
```

This is the expected shape, and it is informative: the behavioural aggregates
are stable, while the time-windowed features have moved because the scoring
population is four months further on. Read it as *"retrain on recent data"*,
not as *"the data is broken"* — a data fault shows up in the data-quality
check, not here.

### Performance monitoring

Performance can only be measured where labels exist, so the job evaluates the
most recent cutoff whose 90-day window has **closed**. Customers in the live
scoring population have no label yet, by construction — which is exactly why
the model is useful for them and useless for measuring itself.

---

## 8. Retraining criteria

Retraining is triggered when **any** of these fire:

1. **Significant drift** — any feature with PSI ≥ 0.25.
2. **Performance below floor** — precision or recall < 0.55, or ROC-AUC < 0.65
   on the most recent closed snapshot.
3. **Input data quality failure** — any domain violation or unexpected nulls.
4. **Staleness** — the model is more than 30 days old.
5. **Bootstrap** — no model artifact exists.

The monitoring report ends with `retrain_recommended` and the explicit list of
reasons, so the decision is auditable rather than a judgement call:

```json
{
  "status": "action_required",
  "retrain_recommended": true,
  "alerts": ["drift: tenure_days PSI 0.5282 >= 0.25"]
}
```

### The automated loop

`dags/churn_ml_dag.py` runs weekly:

```
check_model_health -> should_retrain -> retrain_model -> score_customers -> verify_after_retrain
                           |
                           +-- short-circuits when the model is healthy
```

The DAG deliberately does **not** retrain on every run. A `ShortCircuitOperator`
skips the retrain when no criterion fires, because retraining a healthy model
on a schedule is how a working model quietly gets replaced by a worse one. The
final task re-runs monitoring against the new model: drift against a freshly
rebuilt baseline should be near zero, so anything still alerting there is a
genuine upstream data problem and is logged as a warning rather than accepted.

For a scheduled job outside Airflow, `--fail-on-alert` exits non-zero when a
criterion fires:

```bash
python -m ml.monitoring --fail-on-alert
```

---

## 9. Lifecycle summary

```
transactions (staging.sales_valid)
      |
      v
ml/features.py      snapshot features + inactivity-window label
      |
      v
ml/train.py         3 candidates -> validation selection -> threshold tuning -> test
      |                     |
      |                     +--> MLflow: params, metrics, artifacts
      |                     +--> registry: retail-churn-classifier vN
      v
artifacts/          churn_model.joblib, churn_metrics.json, churn_model_card.json,
                    churn_training_features.csv (the drift baseline)
      |
      +--> ml/registry.py --> api/main.py        (online, /predict)
      +--> ml/registry.py --> ml/score_churn.py  (batch, churn_scores.csv)
      +--> ml/registry.py --> ml/monitoring.py   (drift / performance / latency)
                                     |
                                     v
                              retrain_recommended --> dags/churn_ml_dag.py
```

A model card is written to `artifacts/churn_model_card.json` on every run,
recording intended use, the label definition, the eligible population,
feature importances, metrics, monitoring thresholds, limitations, ethical
considerations and the environment versions.

---

## 10. Limitations

- **Observational transaction data only.** No marketing contact history, no
  demographics, no competitor activity — so the model describes behavioural
  patterns preceding inactivity, not causes of it.
- **Account-level churn.** A customer who buys through a different account or
  channel reads as churned.
- **Two seasonal cycles.** The dataset covers Dec 2009 – Dec 2011, so
  seasonality is learned from two Christmas periods. The November/December
  spike is a large part of what the time-windowed features see.
- **Probabilities are ranking scores, not calibrated risks.** Brier score is
  logged, but the output should not be multiplied by customer value to compute
  an expected loss without calibration first.
- **Threshold is an F1 optimum, not a business optimum.** See §4.
- **Majority-class churn.** At a 90-day window the churn rate is ~58%, so
  accuracy is a weak metric here; ROC-AUC, precision and recall are the ones
  to read.

## 11. Ethical considerations

- No personally identifying attributes are used. `customer_id` is a
  pseudonymous key and no demographic enrichment is joined in.
- Scores are intended to **prioritise retention outreach**. They must not be
  used to vary pricing per customer, which would mean charging the least loyal
  customers differently on the basis of a model they cannot see.
- The decision threshold should be reviewed with whoever runs the campaign
  before any outreach, since it sets how many contented customers get
  contacted unnecessarily.
- The 90-day window is a modelling choice, not a fact about customers. It is
  recorded in the artifact and the model card so any downstream claim about
  "churn" can be traced to the definition that produced it.

---

## 12. Bugs found bringing the stack up in Docker

Running every service from a clean volume, and later from deleted images and
volumes, surfaced seven defects that no unit test could have caught. They are
recorded here because each one is a different class of failure.

- **`db/init/01_init.sh` had CRLF line endings.** Postgres's entrypoint ran it
  and got `/bin/bash^M: bad interpreter`, so the whole database init silently
  never ran — no `retail_dw` database, no schemas. Fixed by converting the file
  to LF and adding `.gitattributes` so Git cannot reintroduce it on a Windows
  checkout.
- **SQLAlchemy 2.1 against mlflow 2.22.** The MLflow server crashed on boot
  looking for `FallbackAsyncAdaptedQueuePool`, which SQLAlchemy 2.1 removed.
  Pinned `SQLAlchemy>=2.0,<2.1` until mlflow catches up.
- **The Airflow image ships pandas 2.1.4.** `churn_ml` died on the `"ME"`
  month-end frequency alias, which only exists from pandas 2.2. The obvious fix
  — bumping the Airflow pin to `pandas>=2.2,<3` — fixed the alias and broke
  every database call instead; see the next entry for how both ended up
  resolved.
- **pandas >= 2.2 against Airflow's `SQLAlchemy<2.0`.** Airflow 2.10.4 pins
  SQLAlchemy below 2.0 (1.4.54 installed), but pandas >= 2.2 requires
  SQLAlchemy >= 2.0. Unable to import a SQLAlchemy it accepts, pandas silently
  fell back to treating a SQLAlchemy `Engine` as a raw DBAPI connection, and
  every `read_sql` / `to_sql` call failed with `AttributeError: 'Engine' object
  has no attribute 'cursor'`. The symptom was `retail_etl`'s
  `validate_and_stage` task failing with the warehouse left empty — a
  dependency conflict presenting as a data bug. Fixed from the other end:
  `_resolve_frequency()` in `ml/features.py` maps both `"M"` and `"ME"` onto
  `pd.offsets.MonthEnd()`, which is valid in every pandas version, so the
  project no longer needs pandas >= 2.2 at all, and both `airflow-webserver`
  and `airflow-scheduler` are pinned to `pandas>=2.1,<2.2`. `ml/config.py`
  still records `SNAPSHOT_FREQUENCY = "ME"`, so the value written to artifacts
  and logged to MLflow runs is unchanged, and the offset produces the same 15
  snapshot cutoffs as the literal `"ME"` did on pandas 2.2.3.
- **MLflow prints an emoji.** On a Windows console that is not UTF-8 the print
  raised `UnicodeEncodeError`, which skipped **all** experiment logging and
  model registration while the training run itself still reported success.
  `ml/train.py` now forces UTF-8 stdout.
- **`retail_etl` allowed concurrent runs.** Two overlapping runs each passed
  `extract`'s "already loaded?" check before either had written its marker, so
  every landing file was ingested twice. Fixed with `max_active_runs=1`.
- **The `api` and `dashboard` images had no `dill`.** `churn_ml`'s
  `retrain_model` task writes `artifacts/churn_model.joblib` from inside the
  Airflow image, which ships dill 0.3.1.1, so the pickle carries dill
  references. The `api` and `dashboard` images are slim `python:3.11-slim`
  builds, so loading that artifact failed with `ModuleNotFoundError: No module
  named 'dill'` and `/health` reported `{"status": "degraded",
  "model_loaded": false}` — the service came up and served the degraded path
  rather than failing loudly. Fixed by adding `dill>=0.3,<1` to
  `requirements.txt` and rebuilding both images.

The pattern worth noting across all seven: four of them failed *quietly*. The
init script, the emoji, the pandas fallback and the missing `dill` each left a
service reporting success while doing nothing, or less than it claimed. That is
the argument for the end-to-end verification counts in the README rather than
trusting a green task list.
