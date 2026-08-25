<div align="center">

# 🦷 Dentiligence — MLOps Practitioner Track

**Production Machine Learning Engineering, from Notebook to Observable System**

*Built by a dentist learning to ship models like an engineer.*

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Coverage](https://img.shields.io/badge/coverage-%E2%89%A570%25-brightgreen)](pyproject.toml)
[![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?logo=docker&logoColor=white)](docker/Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Course](https://img.shields.io/badge/Course-MLOps%20Practitioner%20%C2%B7%20ITI%20%C3%97%20MLOps%20MENA-6f42c1)](https://github.com/MLOpsMENACommunity/mlops_practitioner_course)

</div>

---

## About this repository

This repository is the engineering portfolio for **[Dentiligence](#)**'s founder as she works through the
**MLOps Practitioner** track (ITI × MLOps MENA Community). It follows the course's five-module,
five-pull-request structure exactly, and is written and organized to the standard of a graduate
systems course lab repo (in the spirit of MIT 6.markdown / Stanford CS329S / Harvard CS107 problem-set
repositories): one clean commit history, one reproducible environment, one README that gets a
stranger to a working prediction in three commands.

> **A note on scope.** Per the course handbook, the five mini projects are deliberately built on a
> shared, boring reference model (NYC TLC ride-duration regression) — "the point is the engineering
> around the model, not the model." Dentiligence's own model (Arabic-language dental triage /
> radiograph model) is reserved for the **Final Project**, where it gets dropped into the exact
> infrastructure built here. Building the plumbing on a disposable dataset first, then swapping in
> the real model once, is the whole professional argument for doing it this way.

## Roadmap — five modules, five pull requests

| # | Module | Status | Branch → Tag | What it proves |
|---|--------|:------:|---------------|-----------------|
| 00 | Environment setup | ✅ | — | Repo, Docker, accounts |
| 01 | **Notebook → production service** | ✅ *in this repo* | `module-1-packaging` → `v0.1.0` | Packaging, FastAPI, Docker, tests |
| 02 | The pipeline that retrains itself | ⏳ planned | `module-2-tracking-automation` → `v0.2.0` | MLflow, DVC, CI/CD, Terraform |
| 03 | Serve it three ways, release it safely | ⏳ planned | `module-3-serving-release` → `v0.3.0` | Airflow, BentoML, Locust, canary |
| 04 | The optimization journey | ⏳ planned | `module-4-optimization` → `v0.4.0` | Pruning, quantization, distillation |
| 05 | The observable ML system | ⏳ planned | `module-5-observability` → `v0.5.0` | Prometheus, Grafana, drift, Langfuse |
| 🎓 | **Final Project** | ⏳ planned | `final-project` | Dentiligence's real model, same stack |

This repository currently ships **Module 0 + Mini Project 1**. Modules 2–5 land as separate pull
requests, each tagged, each reviewed, following the course's collaboration protocol.

## Quickstart — 3 commands

```bash
git clone https://github.com/<your-username>/dentiligence-mlops-practitioner.git
cd dentiligence-mlops-practitioner && pip install -e ".[dev]"
uvicorn prodml.api.main:app --reload --port 8000
```

Then:

```bash
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PU_DO": "43_151", "trip_distance": 3.2}'
```

Or with Docker, zero clone required once published:

```bash
docker run --rm -p 8000:8000 <your-dockerhub-user>/prodml-api:0.1.0
```

## Repository layout

```
dentiligence-mlops-practitioner/
├── README.md                  # this file
├── pyproject.toml             # package metadata, deps, entry points
├── .pre-commit-config.yaml    # ruff, black, hygiene hooks
├── notebooks/
│   └── 00-baseline.ipynb      # the messy "before" picture — Step 1
├── src/prodml/                # the installable package
│   ├── config.py              # pydantic-settings, no hardcoded paths
│   ├── logging_conf.py        # structured JSON logging + correlation IDs
│   ├── data.py                # load + split
│   ├── features.py            # feature engineering
│   ├── train.py                # fit + persist (entry point: prodml-train)
│   ├── predict.py             # DurationPredictor — the seam for M3/M4/M5
│   ├── export.py              # pickle → ONNX, with a parity check
│   └── api/
│       ├── schemas.py         # Pydantic request/response models
│       └── main.py            # FastAPI: /health /metadata /predict /predict/batch
├── tests/                     # pytest — fixtures, mocks, ≥70% coverage gate
├── docker/                    # multi-stage Dockerfile + compose
├── reports/
│   └── module-1.md            # MAE, latency, image size, maturity self-assessment
└── docs/
    └── roadmap.md             # the full 5-module + final-project plan
```

## The five commands this project runs on

```bash
pip install -e ".[dev]"                                       # install
ruff check src tests && black --check src tests                # lint
pytest -v --cov=src/prodml --cov-report=term-missing            # test
python -m prodml.train                                          # train
uvicorn prodml.api.main:app --reload --port 8000                 # serve
```

## License

MIT — see [LICENSE](LICENSE). Built for the ITI × MLOps MENA Community "MLOps Practitioner" course.
