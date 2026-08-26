---
tags: [mlops, session1, maturity-model, anti-patterns]
up: "[[00 - MLOps S1 - From Code to Container]]"
---

# Topic 01 · MLOps Maturity Model & the "Everything Notebook"

> [!danger] Correction from an earlier version
> My first pass compressed this into a vague 4-problem list from memory. The **real** `bad_notebook_example.ipynb` has **13 distinct, numbered problems**, each marked with a `❌` comment in its own cell. This matters — if you're asked to spot problems in a notebook (a real interview format, per the README), 4 generic issues is not the same skill as spotting 13 specific ones. Rebuilt in full below.

## MLOps vs DevOps vs DataOps

| | Focus | Goal |
|---|---|---|
| **DevOps** | Shipping software reliably | Build → test → deploy code |
| **DataOps** | Data pipelines | Get data from source → consumer, reliably & fast |
| **MLOps** | Combines both + the model | Productionize models reliably, reproducibly, at scale |

## The Five Levels

| Level | Name | Description |
|---|---|---|
| **0** | No MLOps | Notebooks only. Manual everything. No reproducibility. |
| **1** | Manual Process | Scripts exist but deployments are manual. No tracking. |
| **2** | ML Pipeline | Automated training. Experiment tracking. Basic CI. |
| **3** | CD for ML | Auto retraining on trigger. Model registry. Monitoring. |
| **4** | Full MLOps | Zero-touch pipelines. Auto rollback. Full observability. |

---

## The "Everything Notebook" — `bad_notebook_example.ipynb`, all 13 real problems

The notebook is titled *"my notebook FINAL v3 (USE THIS ONE!!)"* — the filename itself is the first joke: this is "the classic starting point of most ML projects, and the reason most of them never reach production" (README, verbatim). Below are all 13 problems, each with the **actual cell content** and why it matters.

### 1 — Everything in one notebook
> Can't reuse, schedule, or deploy any single piece independently. EDA, preprocessing, training, evaluation, and "tests" are all welded into one file.

### 2 — Unpinned `!pip install` — irreproducible environment
```python
!pip install pandas numpy scikit-learn matplotlib seaborn
```
No versions pinned. Whoever runs this next month gets different library versions — "works on my machine" guaranteed. There's no `pyproject.toml`/lockfile; the environment lives only in this one cell. **Fix**: [[02 - Python Packaging and Project Structure]].

### 3 — Hardcoded secrets
```python
MLFLOW_TRACKING_TOKEN = "sk-live-8f2a9b1c-SUPER-SECRET-do-not-share"
DB_PASSWORD = "admin123"
```
One `git push` away from a credential leak. Should come from environment variables or a secret manager.

> [!warning] Real risk, not hypothetical
> Bots/agents crawl public GitHub repos continuously looking for exactly this pattern in commits, then abuse the leaked credentials. Always list `.env` in `.gitignore` — never commit it.

### 4 — Hardcoded absolute path
```python
df = pd.read_csv("/Users/aya/Desktop/data/taxi_rides_FINAL_v3_use_this.csv")
```
Breaks on every other machine and in CI. Paths belong in config/env vars; data belongs in versioned storage (DVC/GCS).

### 5 — Silent failure that fabricates data
```python
try:
    df = pd.read_csv("/Users/aya/Desktop/data/taxi_rides_FINAL_v3_use_this.csv")
except:
    df = pd.DataFrame({
        "distance": np.random.uniform(0.5, 30, 1000),
        "passengers": np.random.randint(1, 5, 1000),
        "pickup_hour": np.random.randint(0, 24, 1000),
    })
    df["duration"] = df["distance"] * 2.1 + df["passengers"] * 1.5 + np.random.normal(0, 3, 1000)
    print("couldnt find the file, using random data instead lol")
```
> [!danger] This is the single most dangerous cell in the notebook
> If the file is missing, the notebook **quietly trains and ships a model on entirely fake, randomly-generated data** — with a `print()` as the only trace, easy to miss in a long-running notebook. A real pipeline should **fail loudly**, not silently substitute garbage and continue. This cell also has no random seed, compounding the problem — different fake data every run.

### 6 — No random seed anywhere
Same cell as above, and also in the train/test split (#9 below). Nobody can reproduce today's exact model tomorrow.

### 7 — Non-idempotent preprocessing cell
```python
df["distance"] = df["distance"] * 1.60934   # convert to km (i think the data is miles?)
```
> [!danger] Run this cell twice, get a different answer
> This mutates `df` **in place**. Run the cell once: correct km conversion. Run it again (e.g. after fixing a bug two cells up and re-running from here): the distance gets converted **twice**, silently corrupting every downstream number. This is the classic "hidden notebook state" bug — results depend on how many times, and in what order, you happened to execute cells. Also note the comment itself: *"i think the data is miles?"* — an unverified assumption baked directly into a transformation.

### 8 — Magic numbers with no config
```python
df = df[df["duration"] < 120]
df = df[df["distance"] > 0.3]
```
Hardcoded inline, no config file, no way to change per environment, no record of which values produced which model.

### 9 — No `random_state` in the train/test split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```
A different split — and therefore different metrics — every single run. Nobody can ever reproduce the model trained today.

### 10 — Hyperparameter history lost by hand-editing
```python
model_new = RandomForestRegressor(n_estimators=100)   # no seed here either
model_new.fit(X_train, y_train)

# model3 = Lasso(alpha=0.01)          # ❌ commented-out code as "version control"
# model3.fit(X_train, y_train)        # this is exactly what git is for

best_model = model_new   # rf was best when I ran it on tuesday
```
Tried `n_estimators=50, 200, 300` by hand-editing this line and re-running. Which value produced yesterday's "good" model? Nobody knows. And commenting out an old attempt (`# model3 = Lasso...`) as a substitute for version control is exactly the problem git already solves.

### 11 — No experiment tracking
```python
# rf tuesday: 0.97
# rf monday: 0.95 (or 0.94?)
# lr: 0.81
```
Metrics exist only as `print()` output and comments — including a comment that isn't even sure of its own value ("0.95 (or 0.94?)"). No history, no comparison between runs, no record of which data/params/code produced which score. **This is exactly what MLflow solves** (Session 2).

### 12 — Training/serving skew
```python
new_rides = pd.DataFrame({
    "distance": [5.2, 12.0], "passengers": [1, 3], "pickup_hour": [9, 18],
})
# reuses the notebook-global `scaler` — works in this kernel, impossible to
# reproduce in an API process that didn't run this notebook.
new_scaled = scaler.transform(new_rides[features])
```
> [!danger] The miles→km conversion is missing here
> The preprocessing in this "serving" cell is **copy-pasted from training, but subtly different** — the miles→km conversion (problem #7) never happens for `new_rides`. The model now sees differently-scaled inputs at "prediction time" than it saw during training, and **nothing warns you**. The fix: one shared preprocessing function, imported by both training and serving — never copy-paste the same logic twice.

### 13 — Model "versioned" by filename, scaler never saved
```python
pickle.dump(best_model, open("/Users/aya/Desktop/model_final_v2_REAL_final.pkl", "wb"))
print("saved!!")
```
No model registry, no metadata (which data/params/metrics/code produced this file?). And critically: the **fitted `scaler` is never saved** — whoever loads this model later cannot reproduce the exact features it expects. Same trap as problem #12, at save time instead of predict time.

### Bonus problems (not in the README's numbered list, but present in the notebook)

**Fake tests that test nothing:**
```python
assert True
print("test 1 passed ✅")

try:
    assert len(X_test) > 100000   # obviously false
except:
    pass
print("all tests passed ✅✅✅")
```
`assert True` always passes and verifies nothing. The `try/except: pass` pattern **swallows a genuine failure** and reports success anyway — arguably worse than no test at all, since it actively lies.

**Out-of-order cell dependency:**
```python
print("final report:", REPORT_TITLE)   # ← uses REPORT_TITLE...
```
```python
REPORT_TITLE = "ride duration model - final report v3"   # ...defined in the NEXT cell
```
This only works if you already ran the cell *below* it first. "Restart & Run All" crashes with a `NameError`. A notebook like this can never be scheduled or automated — it needs a human who knows the secret cell-execution order.

---

## From notebook to production — the README's fix table

| Notebook problem | Production fix in this repo |
|---|---|
| Everything in one `.ipynb` | Logic extracted into importable modules: `src/model.py` — see [[02 - Python Packaging and Project Structure]] |
| `!pip install` with no versions | Declared, versioned dependencies in `pyproject.toml` |
| Manual cell-based "tests" | Real `pytest` unit tests in `tests/test_model.py` — see [[06 - Structured Logging and pytest]] |
| No entry point | A served API — see [[03 - FastAPI for ML Inference]] |
| Hardcoded paths & "works on my machine" | Containerized with the `Dockerfile`; model path injected via `MODEL_PATH` — see [[05 - Docker and Containerization]] |
| Metrics in print statements & comments | MLflow tracking server in the compose stack (Session 2 builds this out further) |
| Pickle on a Desktop | Models mounted from a versioned `models/` directory |

**The general recipe** (README, verbatim): extract pure functions (load → preprocess → train → evaluate) out of the notebook into modules, parameterize every path and magic number, pin the environment, test the functions with pytest, track experiments instead of printing them, and keep notebooks only for what they're good at — exploration and reporting.

> [!tip] Notice what this recipe does NOT claim
> It doesn't claim this session's repo fixes *all 13* problems completely — several (real experiment tracking, a real model registry, a lockfile) are explicitly still open. See [[09 - What This Session Deliberately Leaves Out]] for the honest list of what's next.
