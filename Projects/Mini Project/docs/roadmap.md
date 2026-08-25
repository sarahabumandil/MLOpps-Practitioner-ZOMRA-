# Roadmap — Dentiligence MLOps Practitioner Track

Source: *The MLOps Practitioner — Mini Projects & Final Project Handbook* (ITI × MLOps MENA
Community). This document maps every module to what lands in this repository, in order.

## Reference model, all mini projects

All five mini projects share one deliberately boring reference model — NYC TLC green-taxi
ride-duration regression — so the engineering, not the modelling, is what's being learned. This
repo's `src/prodml` package implements it; when no real Parquet file is present under
`data/raw/`, `prodml.data` falls back to a deterministic synthetic dataset with the same schema
so every command in this README still runs end to end.

## Module 0 — Environment setup ✅

- [x] GitHub repo, Docker Hub account, SSH keys
- [x] Repository contract (`src/`, `tests/`, `docker/`, `reports/`, `docs/`)
- [x] No Makefile — real commands live in the README

## Mini Project 1 — Notebook → production service ✅ *(this repo, `v0.1.0`)`

- [x] `notebooks/00-baseline.ipynb` — the messy "before"
- [x] `src/prodml/` — packaged, typed, `DurationPredictor` seam
- [x] Structured JSON logs + correlation IDs (`logging_conf.py`)
- [x] pickle → ONNX export with a parity test (`export.py`)
- [x] FastAPI: `/health` `/metadata` `/predict` `/predict/batch`
- [x] pytest suite: fixtures, mocks, parametrization, 70% coverage gate
- [x] Multi-stage Docker image, non-root, `docker-compose.yml`
- [x] `reports/module-1.md` — MAE, latency, image size, maturity self-assessment

## Mini Project 2 — The pipeline that retrains itself ⏳

- [ ] MLflow (Postgres + MinIO) tracking 3 model families, ≥ 13 runs
- [ ] Model Registry: `None → Staging → Production`, `predict.py` loads by stage
- [ ] DVC-versioned data + `pipelines/dvc.yaml` (`prepare → featurize → train → evaluate`)
- [ ] GitHub Actions: lint → test → model-quality gate → build → push
- [ ] Terraform-provisioned artifact store (cloud or Docker-provider path)
- [ ] Continuous training workflow — schedule / manual / `repository_dispatch` triggers,
      human approval gate before Production
- [ ] `reports/module-2.md`, branch `module-2-tracking-automation` → `v0.2.0`

## Mini Project 3 — Serve it three ways, then release it safely ⏳

- [ ] Airflow DAG: extract → train → evaluate → register, with branching + XCom + retries
- [ ] Written inference-pattern decision (web / batch / streaming) before any tool is chosen
- [ ] CAT 1 FastAPI baseline benchmarked; CAT 2 BentoML with micro-batching
- [ ] Batch scoring ≥ 1M rows; streaming via Redis Streams consumer groups
- [ ] ONNX Runtime + OpenVINO (mandatory); TensorRT/Triton or vLLM if hardware allows
- [ ] Locust to 100 users, bottleneck diagnosed from evidence
- [ ] nginx canary release with automatic rollback + shadow mode
- [ ] `reports/module-3.md`, branch `module-3-serving-release` → `v0.3.0`

## Mini Project 4 — The optimization journey ⏳

- [ ] Benchmark harness (warmup, CUDA sync, fixed seeds) — Row 0 = Module 3's numbers
- [ ] Pruning sweep (30/50/70%) with fine-tuning; PTQ vs QAT; knowledge distillation
- [ ] TensorRT FP16/INT8 (or documented Colab path); ONNX Runtime + OpenVINO across providers
- [ ] Edge (TFLite/GGUF) and/or LLM quantization (AWQ/GPTQ) if the final project needs it
- [ ] Journey table + decision framework + stopping rule; winner redeployed into Module 3's stack
- [ ] `reports/module-4.md`, branch `module-4-optimization` → `v0.4.0`

## Mini Project 5 — The observable ML system ⏳

- [ ] Drift simulator (4 types × 4 dynamics) + 6 detection methods implemented and unit-tested
- [ ] Prometheus metric contract, `/metrics`, exporter zoo, correct under multiple workers
- [ ] Grafana dashboard provisioned as code (4 rows), deploy annotations
- [ ] ≥ 6 Alertmanager rules, symptom vs cause, one SLO burn-rate alert
- [ ] Evidently → PostgreSQL → Grafana → CI gate → Airflow retrain branch (closed loop)
- [ ] LLM observability (Langfuse) if the final project has a generative component
- [ ] `docs/runbook.md`, `reports/module-5.md`, branch `module-5-observability` → `v0.5.0`

## Final Project — Dentiligence, on the same infrastructure 🎓

The exact system built above, with Dentiligence's own model swapped in behind the same
`DurationPredictor`-shaped seam — same packaging, same tracking, same serving, same
optimization pipeline, same observability stack. Track and scope to be defined once Mini
Projects 1–5 are merged.
