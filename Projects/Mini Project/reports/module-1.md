# Module 1 — From notebook to production-ready service

**Repository:** dentiligence-mlops-practitioner · **Branch:** `module-1-packaging` · **Tag:** `v0.1.0`

## 1. Baseline vs packaged — validation metrics

| | MAE (min) | RMSE (min) |
|---|---|---|
| `notebooks/00-baseline.ipynb` | _fill in after running_ | _fill in_ |
| `python -m prodml.train` | _fill in — must match baseline within ±0.05_ | _fill in_ |

> Run `jupyter nbconvert --to notebook --execute notebooks/00-baseline.ipynb` and
> `python -m prodml.train`, then paste both MAE numbers here.

## 2. Serialization: pickle vs ONNX

| Format | Human-readable | Cross-language | Schema-enforced | Safe from untrusted source |
|---|---|---|---|---|
| JSON | ✅ | ✅ | ❌ | ✅ |
| Protobuf | ❌ | ✅ | ✅ | ✅ |
| Pickle | ❌ | ❌ | ❌ | ❌ — executes arbitrary code on load |
| ONNX | ❌ | ✅ | ✅ | ✅ |

**Decision:** the service serves pickle internally (trusted, self-produced artifacts only,
never loaded from an untrusted source) and exports ONNX as the portable, cross-language
artifact for later serving categories (Module 3/4). Never load a `.pkl` you did not produce.

| | Mean latency (ms) | p95 latency (ms) |
|---|---|---|
| Pickle | _fill in from `pytest -k serialization -s` or a manual benchmark_ | _fill in_ |
| ONNX Runtime | _fill in_ | _fill in_ |

Parity test: `tests/test_serialization.py::test_pickle_onnx_parity` — max abs diff over 50–500
samples: _fill in_ (must be < 1e-3).

## 3. Docker image size — single-stage vs multi-stage

| Build | Image size |
|---|---|
| Single-stage (no builder split) | _fill in_ |
| Multi-stage (`docker/Dockerfile` as shipped) | _fill in_ |

```bash
docker images | grep prodml-api
```

## 4. MLOps maturity self-assessment

Placing this repository on the five-level maturity model from Lesson 1:

- **Level reached:** _fill in (e.g. "Level 1 — DevOps but no MLOps": packaging, testing, CI and
  containerization are in place; there is no experiment tracking, model registry or automated
  retraining yet.)_
- **What's missing to reach the next level:** experiment tracking (MLflow), a model registry, and
  data versioning (DVC) — all of which is Module 2.

## 5. Definition of Done

- [ ] GitHub repo public, more than one commit; Docker Hub token saved securely
- [ ] `docker run hello-world` works
- [ ] `pip install -e .` succeeds in a clean virtualenv
- [ ] Lint passes; pre-commit hooks installed
- [ ] Tests pass with coverage ≥ 70%
- [ ] Zero `print()` statements in `src/`
- [ ] `/health`, `/metadata`, `/predict`, `/predict/batch` all respond correctly
- [ ] ONNX parity test passes
- [ ] Image pushed to Docker Hub and pullable by someone else; container runs as non-root
- [ ] `README.md` gets a stranger to a prediction in 3 commands
- [ ] This report has: MAE, latency comparison, image-size comparison, serialization table,
      maturity self-assessment
- [ ] Pull request opened, reviewed by two peers, merged; `v0.1.0` tagged
