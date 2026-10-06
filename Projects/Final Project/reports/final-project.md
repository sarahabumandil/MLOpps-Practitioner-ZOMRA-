# Final Project report — arasent (Track 1: Arabic Sentiment Analysis)

## Honesty clause

Per the handbook's own honesty clause: numbers below are exactly what was measured in this
environment (no GPU, no internet to HuggingFace/Kaggle), on synthetic-but-schema-faithful
data where the real Kaggle dataset wasn't reachable. Where a number needs a real dataset or
a GPU to be meaningful, that's stated explicitly rather than estimated.

## 1. Code & packaging ✅

`pip install -e ".[dev]"` succeeds in a clean venv. `SentimentClassifier` (baseline) and
`ArabertSentimentClassifier` (real Track-1 model) share one interface — verified by
`tests/test_model.py` and `tests/test_predict.py` (backend-selectable loader).

Test suite: **21/21 passed, 83% coverage** (gate: 70%).

## 2. API ✅

Verified live (`uvicorn` + `curl`):

| Endpoint | Result |
|---|---|
| `GET /health` | `{"status":"healthy","model_loaded":true}` |
| `GET /metadata` | `{"model_version":"0.1.0-baseline","framework":"scikit-learn (TF-IDF + LogisticRegression)",...}` |
| `POST /predict` (positive review) | `{"label":"positive","confidence":0.9549,...}` |
| `POST /predict` (negative review) | `{"label":"negative","confidence":0.9634,...}` |
| `POST /predict` empty text | `422` |
| `GET /metrics` | Prometheus exposition format, `predictions_total` and `prediction_latency_seconds` present |

## 3. Docker

Multi-stage `docker/Dockerfile` + `docker-compose.yml` (API + MLflow + Prometheus + Grafana).
Docker itself is not available in this build environment, so image-size numbers could not be
measured here — same caveat as Mini Project 1. Run locally:

```bash
docker build -f docker/Dockerfile -t <user>/arasent-api:0.1.0 .
docker images | grep arasent-api
```

## 4. MLflow tracking ✅ — real run

`python -m arasent.train` — 5-run sweep over `(C, max_features)`, every run logged with
params/metrics/artifacts/tags, best run registered as `ArabicSentiment` and promoted to
**Production** in the Model Registry (`MlflowClient.transition_model_version_stage`).

```json
{
  "best_run_id": "c87e039ff6ae4da7ad74f5ffbef211ac",
  "accuracy": 1.0,
  "f1_macro": 1.0,
  "registered_version": 2
}
```

(Accuracy/F1 of 1.0 reflects the small, easy synthetic/placeholder dataset used to prove the
pipeline — expect real numbers in the 0.75–0.90 range on the actual 100k-review Kaggle
dataset, consistent with published AraBERT sentiment baselines.)

## 5. DVC data versioning ✅ — real run

`pipelines/dvc.yaml`: `prepare → train → evaluate`. Verified with `dvc repro`:

- First run: all 3 stages executed, `dvc.lock` generated.
- Second run (unchanged): **all 3 stages skipped** (`"Stage 'prepare' didn't change, skipping"`)
  — proves caching works.
- `dvc metrics show`:

  | Path | accuracy | f1_macro |
  |---|---|---|
  | `reports/train_metrics.json` | 1.0 | 1.0 |
  | `reports/eval_metrics.json` | 1.0 | 1.0 |

To reproduce with the real dataset: `dvc add data/raw/arabic_reviews.csv` after downloading
it, configure a DVC remote (`dvc remote add -d storage <url>`), then `dvc repro`.

## 6. CI/CD

`.github/workflows/ci.yml` — validated as syntactically correct YAML. Jobs: `lint → test →
quality-gate (trains and checks f1_macro ≥ 0.60, fails the build otherwise) →
build-and-push (Docker Hub, main branch only)`. Not executed here since it requires GitHub's
runners and repo secrets — push to GitHub and open a PR to see it run.

## 7. Production serving ✅ — real run

**BentoML** (`serving/service.py`, BentoML 1.4 decorator API, `@bentoml.api(batchable=True,
max_batch_size=32, max_latency_ms=200)`): served live, verified:

```json
// POST /health
{"status": "healthy", "labels": ["negative","neutral","positive"], "model_version": "0.1.0-baseline"}
// POST /predict_one {"text": "المنتج رائع جدا"}
{"label": "positive", "confidence": 0.893}
```

**Locust** (`loadtest/locustfile.py`, realistic 80% single / 15% batch / 5% metadata mix,
30 concurrent users, 20s, against the FastAPI baseline):

| Endpoint | Requests | Failures | Avg | p50 | p95 | Max |
|---|---|---|---|---|---|---|
| `POST /predict` | 247 | 0 | 6ms | 4ms | 24ms | 57ms |
| `POST /predict/batch` | 40 | 0 | 7ms | 4ms | 53ms | 60ms |
| **Aggregated** | 300 | **0 (0%)** | 6ms | 4ms | 24ms | 60ms |

Throughput: ~15 req/s at 30 users on this (constrained, shared) sandbox machine — re-run on
real hardware for production capacity planning; the point proven here is **0% failure rate
under load and the harness itself working end to end**.

## 8. Monitoring ✅

- **Prometheus `/metrics`**: `predictions_total{status,label}` (counter),
  `prediction_latency_seconds` (histogram, SLA-aligned buckets), `prediction_confidence`
  (histogram — the cheapest continuous drift signal) — all verified present and updating
  live via `tests/test_api.py::test_metrics_endpoint_exposes_prometheus_format`.
- **PSI drift** (`monitoring/drift.py`): 3/3 tests pass — near-zero PSI for identical
  distributions, > 0.25 for a clearly shifted one, `drift_report()` correctly flags
  `drifted: true` when either `text_length` or `confidence_score` PSI crosses 0.25.
- **Alerts** (`monitoring/alerts.yml`): error rate, p95 latency, PSI drift, and volume-drop
  rules, each with a `for:` clause and a first-action annotation.
- **Grafana**: 4-panel dashboard provisioned from `monitoring/grafana/provisioning/` (p95
  latency, predictions/sec by label, confidence heatmap, error rate) — not run live here
  (needs the full Compose stack), but provisioning files are in place and reference the
  exact metric names emitted by the API.

## 9. ONNX optimization (baseline) ✅ — real run

```json
{
  "parity_ok": true,
  "max_abs_diff": 6.09e-08,
  "sklearn_latency_ms": {"mean_ms": 0.101, "p95_ms": 0.132},
  "onnx_latency_ms":    {"mean_ms": 0.0075, "p95_ms": 0.0092}
}
```

~13.5x mean-latency speedup, single-row inference, parity confirmed to 1e-3 tolerance.

## 10. AraBERT — the real Track-1 model (GPU required, not run here)

`notebooks/finetune_arabert_colab.ipynb` is complete and syntax-checked (all 8 code cells
parse as valid Python) but requires a GPU runtime + HuggingFace Hub access to execute. It:

1. Loads the Kaggle Arabic-reviews CSV (same column-detection logic as `data.py`).
2. Fine-tunes `aubmindlab/bert-base-arabertv02` across a 5-run `(lr, batch_size)` sweep,
   logged to MLflow.
3. Distills the best run 12→6 layers (temperature-scaled KL + cross-entropy loss).
4. Exports the student to ONNX and quantizes to INT8 via Optimum.
5. Benchmarks all three variants (teacher / student / student+INT8) with a proper warmup +
   50-iteration harness, writing `journey_table.csv`.
6. Documents the exact steps to drop the checkpoint into `models/arabert-sentiment/` and
   flip `ARASENT_MODEL_BACKEND=transformer` — no other code changes.

**Action item before submission:** run this notebook on Colab (free T4 GPU), copy the
resulting `journey_table.csv` into this report, and copy the checkpoint into the repo.

## Peer review

Not applicable to this build session — submit the merged repo for peer review per the
course's collaboration protocol (two reviewers, ≥300-word written review each).
