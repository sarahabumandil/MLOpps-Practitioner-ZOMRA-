---
tags: [mlops, session1, testing, pytest, logging]
up: "[[00 - MLOps S1 - From Code to Container]]"
---

# Topic 06 · Unit Testing with pytest (+ the Logging Gap)

> [!danger] Correction from an earlier version of this note
> I previously wrote a confident `structlog` code example as if structured logging already exists in this project. **It doesn't.** The README states directly: *"Today the app never logs anything."* Structured logging is item #3 on the "🔜 next steps" list, not a built feature. I've moved that content to the bottom of this note, clearly marked as aspirational, and rebuilt the pytest section from the actual `tests/test_model.py` file.

## The testing pyramid (from the README, verbatim structure)
```
        ▲  e2e tests         — few: the whole system, slow, broad
       ▲▲  integration tests — some: pieces working together (e.g. API + model)
      ▲▲▲  unit tests        — many: single functions, fast, precise
```
- A **unit test** checks one small piece of code (typically a single function/method) in isolation: given this input, does it return this output?
- Fast (milliseconds), no network/database/GPU needed, pinpoints exactly which piece broke.
- Contrast with the bad notebook's "tests" (`assert True` in a cell, run by hand, gone on kernel restart) — real tests are **code that lives in the repo**, run by a test runner, executed *automatically* on every push by CI.
- In ML projects specifically, unit tests are what let you refactor preprocessing or swap a model implementation with confidence that the contract (`predict([distance, passengers]) -> minutes`) still holds.

## pytest basics
`pytest` is Python's de-facto test runner:
- Auto-discovers files named `test_*.py`
- Treats every `test_*` function as a test case
- Gives you plain `assert` statements
- **Fixtures** — reusable setup (like a pre-configured model)
- **parametrize** — run one test body over many input/expected pairs

All three appear in the actual `tests/test_model.py`.

## The real `tests/test_model.py` — in full
```python
import pytest
from unittest.mock import MagicMock
from src.model import RideDurationModel


# ── Basic test ─────────────────────────────────────
def test_predict_returns_float():
    model = RideDurationModel()
    model._model = MagicMock()
    model._model.predict.return_value = [23.5]
    result = model.predict([5.0, 1])
    assert isinstance(result, float)
    assert result == 23.5


# ── Parameterized tests ────────────────────────────
@pytest.mark.parametrize(
    "distance,pax,expected",
    [
        (1.0, 1, 5.2),
        (10.0, 2, 24.8),
        (0.5, 4, 3.1),
    ],
)
def test_predict_multiple_inputs(distance, pax, expected):
    model = RideDurationModel()
    model._model = MagicMock()
    model._model.predict.return_value = [expected]
    assert model.predict([distance, pax]) == expected


# ── Fixture for shared setup ───────────────────────
@pytest.fixture
def mock_model():
    m = RideDurationModel()
    m._model = MagicMock()
    m._model.predict.return_value = [15.0]
    return m


def test_threshold_clipping(mock_model):
    mock_model.threshold = 10.0
    result = mock_model.predict([100.0, 1])
    assert result == 10.0  # clipped to threshold
```

### What each test actually verifies

| Test | What it checks | Technique |
|---|---|---|
| `test_predict_returns_float` | `.predict()` returns a plain `float`, and the value matches what the mocked estimator returned | `MagicMock` replaces `model._model` entirely — the *real* estimator's math is never exercised |
| `test_predict_multiple_inputs` | Same behavior across 3 different input/output pairs, without repeating the test body 3 times | `@pytest.mark.parametrize` |
| `test_threshold_clipping` | If `threshold=10.0` and the (mocked) raw prediction is `15.0`, the returned value is clipped down to `10.0` | `@pytest.fixture` builds a ready-to-use mocked model once, reused across any test that asks for the `mock_model` argument |

> [!important] Why mock `_model` instead of testing the real heuristic math
> Mocking the internal estimator keeps these tests focused on `RideDurationModel`'s **own logic** — delegation to `self._model.predict()` and the threshold-clipping behavior — rather than the arithmetic of whichever estimator happens to be plugged in. If you swapped `_HeuristicEstimator` for a real trained model tomorrow, these three tests would still pass unchanged, because they never depend on *how* the number gets computed — only on the fact that `RideDurationModel` correctly delegates and clips.

### Running the tests
```bash
pip install -e ".[dev]"
pytest             # run everything
pytest -v          # verbose: one line per test case
pytest -k thresh   # run only tests matching a keyword (e.g. "thresh" → test_threshold_clipping)
```
Expected output: **`5 passed`**.

> [!note] Why 5, when only 3 test *functions* are shown above
> `test_predict_multiple_inputs` is parametrized over 3 input sets, so pytest counts it as 3 separate test cases. `1 + 3 + 1 = 5`.

## What testing does NOT yet cover here (honest gap, from the README)
The current test file only covers the **model class** (`RideDurationModel`) in isolation. It does **not** test:
- The FastAPI `/predict` or `/health` endpoints themselves (would need `TestClient`)
- The happy-path *and* validation-error paths of the actual HTTP layer
This is explicitly listed as a next step (#4 in the README's list) — see [[09 - What This Session Deliberately Leaves Out]].

---

## ⚠️ Structured logging — NOT built yet in this repo

> [!danger] This is aspirational content, not a description of existing code
> Everything in this section describes what the README recommends **adding next** — it is explicitly **not** in `fastapi_example.py` or `litestar_example.py` today. The README's exact words: *"Today the app never logs anything."* If you go looking for a logging module in this repo, you won't find one — that's correct, not a bug in these notes.

The README's specific recommendation (item #3 on the "🔜 next steps" list):
> Read `LOG_LEVEL` [already set in `docker-compose.yml`'s `environment:` block, but currently unused], log every prediction (inputs, output, latency, model version).

The general shape such a fix would take (illustrative — not repo code):
```python
# Illustrative only — this code does not exist in the repo yet.
import structlog

log = structlog.get_logger()
log.info("predict.made", distance=req.distance, passengers=req.passengers,
          duration_min=duration, latency_ms=4.1)
```
The core idea, if/when you build this yourself as practice: `print()` output isn't searchable or leveled; JSON-structured logs with a consistent shape are what actually get ingested by log aggregation tools in production. But treat this section as a **suggested exercise**, not a description of what to expect when you open this repo's code.
