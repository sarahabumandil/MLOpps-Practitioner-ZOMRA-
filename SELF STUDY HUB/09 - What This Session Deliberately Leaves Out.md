---
tags: [mlops, session1, roadmap]
up: "[[00 - MLOps S1 - From Code to Container]]"
---

# Topic 09 · What This Session Deliberately Leaves Out

> [!info] Why this note exists
> The repo's README ends with an explicit, numbered list of what's *intentionally* not built at this session's scope, and why. This is one of the most useful things to study — it tells you exactly what a reviewer/interviewer might ask "why isn't X here?" about, and gives you the correct answer each time: "it's deliberately out of scope for a single-model-behind-an-API session; here's what would be added next." Several other notes in this vault link back here instead of repeating this list.

## ✅ What this session's code actually demonstrates
- Code as importable modules (`src/model.py`) instead of a notebook
- Declared, versioned dependencies (`pyproject.toml`) with optional extras
- Unit tests with pytest (`tests/`)
- A served model behind a documented API (FastAPI/Litestar + OpenAPI)
- Reproducible, non-root, multi-stage container build (`Dockerfile`)
- Local stack wiring with compose (API + MLflow) and a read-only model mount
- A portable model format (ONNX export)

## 🔜 The 8 next steps, in the README's own order

| # | Gap | What's missing today | The fix |
|---|---|---|---|
| 1 | **Actually load the model from `MODEL_PATH`** | Compose sets the env var and mounts `models/`, but the app always uses the built-in heuristic — see [[03 - FastAPI for ML Inference]] and [[05 - Docker and Containerization]] | Load the pickle at startup via FastAPI's *lifespan*, fall back to the heuristic with a loud log line if missing |
| 2 | **Input validation** | Request fields have no constraints at all (`distance: float` accepts negatives) | Constrain the schema: `distance: Field(gt=0)`, `passengers: Field(ge=1, le=8)` |
| 3 | **Structured logging** | The app never logs anything today — see [[06 - Structured Logging and pytest]] | Read `LOG_LEVEL` (already set in compose, unused), log every prediction: inputs, output, latency, model version |
| 4 | **API tests** | Current tests only cover the `RideDurationModel` class, not the HTTP layer — see [[06 - Structured Logging and pytest]] | Add `TestClient` tests for `/predict` (happy path + validation errors) and `/health` |
| 5 | **A real health check** | `/health` unconditionally returns `"healthy"` regardless of model state | Verify the model is actually loaded; add a Docker/compose `HEALTHCHECK`; add a `/model-info` endpoint (version, source, loaded-at) |
| 6 | **A lockfile** | `pyproject.toml`'s `>=` bounds reproduce the *project*, not a byte-identical *environment* — see [[02 - Python Packaging and Project Structure]] | `uv lock` or `pip-compile` to pin every transitive dependency |
| 7 | **Run this session's tests in CI** | The repo's GitHub Actions workflow currently only tests `session_2/` | A paths-filtered job should run `session_1/` tests on every push too |
| 8 | **Model + preprocessing saved together** | If a scaler/encoder isn't shipped with the model, you get exactly the training/serving skew shown in the bad notebook — see [[01 - MLOps Maturity Model]] | Ship one combined pipeline artifact (model + preprocessing), not the model alone |

## Where later sessions pick this up
Per the README: experiment tracking (Session 2), data/model versioning with DVC, CI/CD to a registry, and serving/scaling (Session 3).

> [!tip] How to use this list while studying
> For each of the 8 rows, try to answer *without looking*: "why does this matter, and what's the one-line fix?" If you can do that for all 8, you understand this session's boundaries correctly — which is arguably more valuable than memorizing the code that *is* there.
