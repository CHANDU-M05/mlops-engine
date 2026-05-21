# mlops-handbook 🚀

A personal reference handbook for MLOps — from experiment tracking to production deployment and monitoring.

Built while working through the full MLOps stack. Each module is a self-contained reference with code, notes, and working examples.

---

## Modules

| # | Module | Tools |
|---|---|---|
| 01 | [Intro & Problem Framing](./01-intro/) | Python, Jupyter, scikit-learn |
| 02 | [Experiment Tracking](./02-experiment-tracking/) | MLflow |
| 03 | [Orchestration & Pipelines](./03-orchestration/) | Prefect |
| 04 | [Model Deployment](./04-deployment/) | FastAPI, Docker, AWS |
| 05 | [Monitoring](./05-monitoring/) | Evidently, Grafana, Prometheus |
| 06 | [Best Practices](./06-best-practices/) | CI/CD, testing, linting |
| 07 | [End-to-End Project](./07-project/) | Full stack |

---

## Stack

- **Experiment tracking** — MLflow
- **Orchestration** — Prefect
- **Deployment** — FastAPI + Docker + AWS Lambda / Kinesis
- **Monitoring** — Evidently + Grafana + Prometheus
- **CI/CD** — GitHub Actions

---

## How to use this

Each module has its own README with:
- Concept summary
- Working code
- Commands to run
- Personal notes

Clone and run any module independently:

```bash
git clone https://github.com/CHANDU-M05/mlops-handbook
cd mlops-handbook/02-experiment-tracking
pip install -r requirements.txt
```

---

## Setup

```bash
# Python 3.10+ recommended
python3 -m venv .venv
source .venv/bin/activate
pip install mlflow prefect evidently fastapi uvicorn
```

---

## License

MIT
