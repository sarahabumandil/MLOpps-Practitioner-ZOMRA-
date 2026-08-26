---
tags: [mlops, session1, fastapi, api, litestar]
up: "[[00 - MLOps S1 - From Code to Container]]"
---

# Topic 03 · From Notebook to REST API (FastAPI + Litestar)

> [!info] Source of truth
> This note is written directly from the repo's actual `fastapi_example.py`, `litestar_example.py`, and `README.md` — not reconstructed from memory of the lecture. Where the code is simpler than you might expect, that's intentional: the README has an explicit "🔜 what's next" list (see [[09 - What This Session Deliberately Leaves Out]]) for everything not yet added.

## Why you need a framework at all
Without a framework you'd build from scratch:
- Raw HTTP/socket handling
- Docs (nobody knows how to call your endpoint)
- Async handling (one user's request could block everyone else's)

**A framework (FastAPI or Litestar) gives you:**
- Request/response schemas via Pydantic
- HTTP handling built-in
- Auto-generated interactive docs — zero extra code
- Async support

## Sync vs async — why it matters
- **Sync (blocking)**: one request must fully finish before the server starts the next.
- **Async**: the server can handle many requests concurrently without waiting for each to finish one-by-one.
- Interview vocabulary trio: **latency**, **throughput**, **concurrency**.

## The actual FastAPI example (`fastapi_example.py`)
This is the real file, in full:
```python
from fastapi import FastAPI
from pydantic import BaseModel
from src.model import RideDurationModel

app = FastAPI(title="Ride Duration API")
model = RideDurationModel()          # ← loaded once at module level


class PredictRequest(BaseModel):
    distance: float
    passengers: int = 1


@app.post("/predict")
async def predict(req: PredictRequest) -> dict:
    duration = model.predict([req.distance, req.passengers])
    return {"duration_min": duration, "status": "ok"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

> [!important] What this example does NOT yet have
> - **No `Field()` constraints** — `distance: float` accepts negative numbers, zero, anything. The README lists proper validation (`Field(gt=0)`, `Field(ge=1, le=8)`) as a **🔜 next step**, not something already built.
> - **No `response_model=`** — the handler just returns a plain `dict`. FastAPI infers the OpenAPI schema from the return type hint (`-> dict`), but there's no dedicated response class.
> - **The model never loads from `MODEL_PATH`** — even though `docker-compose.yml` sets that env var and mounts `models/`, the app code never reads it. It always uses the built-in heuristic in `src/model.py`. This is deliberate, to keep this session's scope small — see [[09 - What This Session Deliberately Leaves Out]].
>
> This isn't a mistake in the example — it's the *starting point* the course builds forward from. Don't copy patterns from other tutorials onto this file expecting them to already be there.

### Running it
```bash
python fastapi_example.py
# or, with auto-reload for development:
uvicorn fastapi_example:app --host 127.0.0.1 --port 8000 --reload
```
Server: **http://127.0.0.1:8000**

### Endpoints (as they actually behave)

| Method | Path | Request body | Response |
|---|---|---|---|
| `POST` | `/predict` | `{"distance": 10, "passengers": 2}` (passengers optional, defaults to 1) | `{"duration_min": 21.0, "status": "ok"}` |
| `GET` | `/health` | — | `{"status": "healthy"}` |

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance": 10, "passengers": 2}'
# → {"duration_min": 21.0, "status": "ok"}
```

### Interactive docs (auto-generated, zero extra code)
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Raw OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

## The Litestar version (`litestar_example.py`) — same model, different pattern
Litestar is a second framework doing the *same job*, included specifically to show a different idiom: **dependency injection**.
```python
from litestar import Litestar, post, get
from litestar.di import Provide
from pydantic import BaseModel
from src.model import RideDurationModel


class PredictRequest(BaseModel):
    distance: float
    passengers: int = 1


class PredictResponse(BaseModel):
    duration_min: float
    status: str = "ok"


def get_model() -> RideDurationModel:
    return RideDurationModel()


@post("/predict")
async def predict(
    data: PredictRequest,
    model: RideDurationModel,      # ← injected, not imported/instantiated in this function
) -> PredictResponse:
    dur = model.predict([data.distance, data.passengers])
    return PredictResponse(duration_min=dur)


@get("/health")
async def health() -> dict:
    return {"status": "healthy"}


app = Litestar(
    route_handlers=[predict, health],
    dependencies={
        "model": Provide(get_model, use_cache=True, sync_to_thread=False),
    },
)
```
| Piece | What it does |
|---|---|
| `Provide(get_model, ...)` | Registers `get_model` as the source of the `model` argument — Litestar calls it and hands the result to any handler that asks for a `model:` parameter |
| `use_cache=True` | Build the model **once**, reuse it for every request — same goal as FastAPI's module-level `model = RideDurationModel()`, achieved through the DI system instead |
| `sync_to_thread=False` | Tells Litestar the factory function is non-blocking, so it shouldn't be offloaded to a worker thread |

> [!note] Same job, two idioms
> FastAPI: instantiate the model as a module-level global, reference it directly inside the handler.
> Litestar: register a factory function, let the framework inject the instance. Functionally equivalent for this simple case — DI becomes more valuable once you have several interdependent services to wire together.

### Running Litestar
```bash
python litestar_example.py
# or:
litestar --app litestar_example:app run --host 127.0.0.1 --port 8001 --reload
```
Server: **http://127.0.0.1:8001** (different port from FastAPI so you can run both side by side)
- Swagger UI: `http://127.0.0.1:8001/schema/swagger`
- ReDoc: `http://127.0.0.1:8001/schema/redoc`

Both apps expose identical `/health` and `/predict` contracts — only the port differs.

## The model being served — `src/model.py`
Both examples call the same underlying model:
```python
class RideDurationModel:
    def __init__(self, threshold: Optional[float] = None) -> None:
        self._model: Estimator = _HeuristicEstimator()   # default: not a trained model
        self.threshold = threshold

    def predict(self, features: list[float]) -> float:
        duration = float(self._model.predict(features)[0])
        if self.threshold is not None:
            duration = min(duration, self.threshold)
        return duration
```
- `_HeuristicEstimator` is **not** a trained ML model — it's `distance / avg_speed + passengers * overhead`, deliberately simple so the session can focus on packaging/serving/testing rather than modeling.
- `self._model` can be swapped for any object with a scikit-learn-style `.predict(features) -> [value]` method — that's the abstraction point. See [[02 - Python Packaging and Project Structure]] for why this matters.
- `threshold` optionally clips the output — used directly in the pytest examples (see [[06 - Structured Logging and pytest]]).

## Why `/health` matters even though it "obviously" works
In production, an orchestrator (Kubernetes or similar) polls `/health` continuously to decide whether to route traffic to this instance — liveness/readiness probes. Not needed for a solo script; essential once deployed behind an orchestrator. The README's "🔜 next steps" list specifically calls out that a *real* health check should also verify the model is loaded — right now it just always returns `"healthy"` unconditionally.

## 422 vs 400 (general FastAPI knowledge, not yet exercised by this example)
Because this example has no `Field()` constraints, sending `distance: -5` today would **not** trigger a 422 — it would happily compute a (nonsensical) prediction. Once validation is added (see [[09 - What This Session Deliberately Leaves Out]]):
- **400** = generic bad request (wrong endpoint, wrong method)
- **422** = request reached the right endpoint, but the *content* failed schema validation — returned automatically by FastAPI before your function body runs
