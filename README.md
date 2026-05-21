# mlops-engine ⚙️

A hands-on personal MLOps reference — built and maintained by me while working through the full MLOps stack end to end.

Each module is self-contained with working code, clear explanations, and commands you can run immediately.

---

## Modules

| # | Topic | Tools |
|---|---|---|
| [01 — Intro](./01-intro/) | Problem framing, baseline model | Python, Jupyter, scikit-learn |
| [02 — Experiment Tracking](./02-experiment-tracking/) | Track runs, params, metrics, models | MLflow |
| [03 — Orchestration](./03-orchestration/) | Build and schedule ML pipelines | Prefect |
| [04 — Deployment](./04-deployment/) | Serve models via API, batch, streaming | FastAPI, Docker, AWS Lambda |
| [05 — Monitoring](./05-monitoring/) | Detect drift, track metrics | Evidently, Grafana, PostgreSQL |
| [06 — Best Practices](./06-best-practices/) | CI/CD, testing, infrastructure | GitHub Actions, Terraform, pytest |
| [07 — Project](./07-project/) | End-to-end reference project | Full stack |

---

## Stack01-intro
---

## Quick Start

```bash
git clone https://github.com/CHANDU-M05/mlops-engine
cd mlops-engine
python3 -m venv .venv
source .venv/bin/activate
pip install mlflow prefect evidently fastapi uvicorn xgboost pandas
```

---

## Module Guides

### 01 — Intro
Baseline NYC taxi trip duration prediction using linear regression.
```bash
cd 01-intro
jupyter notebook duration-prediction.ipynb
```

### 02 — Experiment Tracking
Track experiments with MLflow. Compare runs, register best model.
```bash
cd 02-experiment-tracking
pip install -r requirements.txt
mlflow ui --host 0.0.0.0 --port 5000
jupyter notebook duration-prediction.ipynb
```

### 03 — Orchestration
Full training pipeline as a Prefect flow with logging and artifact saving.
```bash
cd 03-orchestration/code
python duration-prediction.py --year 2023 --month 1
```

### 04 — Deployment
**Web service (FastAPI):**
```bash
cd 04-deployment/web-service
uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID":"130","DOLocationID":"205","trip_distance":3.5}'
```

**Batch scoring:**
```bash
cd 04-deployment/batch
python score.py
```

**Streaming (AWS Lambda + Kinesis):**
```bash
cd 04-deployment/streaming
docker build -t duration-model .
docker run -p 8080:8080 duration-model
```

### 05 — Monitoring
Drift detection with Evidently. Results in PostgreSQL, visualised in Grafana.
```bash
cd 05-monitoring
docker compose up -d
pip install -r requirements.txt
python evidently_metrics_calculation.py
open http://localhost:3000
```

**Env vars:**
```bash
export POSTGRES_HOST=localhost
export POSTGRES_PASSWORD=yourpassword
export REFERENCE_DATA=data/reference.parquet
export CURRENT_DATA=data/green_tripdata_2022-02.parquet
export MODEL_PATH=models/lin_reg.bin
export NUM_DAYS=27
```

### 06 — Best Practices
CI/CD, unit + integration tests, Terraform infrastructure.
```bash
cd 06-best-practices/code
pytest tests/
make quality_checks
cd infrastructure
terraform init
terraform plan -var-file=vars/stg.tfvars
terraform apply -var-file=vars/stg.tfvars
```

---

## Environment Variables

| Variable | Default | Module |
|---|---|---|
| `POSTGRES_HOST` | `localhost` | 05-monitoring |
| `POSTGRES_PORT` | `5432` | 05-monitoring |
| `POSTGRES_DB` | `monitoring` | 05-monitoring |
| `POSTGRES_USER` | `postgres` | 05-monitoring |
| `POSTGRES_PASSWORD` | `postgres` | 05-monitoring |
| `MODEL_PATH` | `models/lin_reg.bin` | 04, 05 |
| `MODEL_BUCKET` | `mlflow-models` | 06 |
| `RUN_ID` | — | 06 |
| `PREDICTIONS_STREAM_NAME` | `ride_predictions` | 06 |
| `TEST_RUN` | `False` | 06 |
| `KINESIS_ENDPOINT_URL` | — | 06 (local) |

---

## License

MIT
