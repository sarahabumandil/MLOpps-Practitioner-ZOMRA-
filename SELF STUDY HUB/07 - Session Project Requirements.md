---
tags: [mlops, session1, project]
up: "[[00 - MLOps S1 - From Code to Container]]"
---

# Topic 07 · Session 1 Project — "Containerized ML API"

> [!warning] Read this before the checklist below
> Aya said this **three separate times, explicitly**: this project is **"اوبشنال جدا"** (very optional) and **"مالوش علاقه بالسيرتفيكيشن خالص"** (has zero connection to either certificate). It's not graded, not reviewed, not submitted anywhere. It exists purely so you can practice tonight's concepts before they get built on next session. Don't over-invest — a simple working version beats a polished unfinished one.
>
> The elaborate checklist below is the **slide deck's idealized version**. What Aya actually asked for out loud was much simpler (see "What to build" just below). If you're short on time, do the simple version and move on.

## What Aya actually asked for (verbal, simple version)
Take **any model you already have** and:
1. Wrap it in a **FastAPI** app (`/predict` endpoint)
2. Build dependencies using **`uv`** (`pyproject.toml`)
3. Write a **Dockerfile** + build a **Docker image**
4. **Push** the image to your Docker Hub account

That's it. Everything past this point is the "if you want to go further" version.

## The fuller checklist (from the slides — optional stretch goal)

| # | Deliverable | Details |
|---|---|---|
| 1 | **Python package** | Clean `src/` layout with `pyproject.toml`. `pip install -e .` works. |
| 2 | **FastAPI service** | `/predict`, `/health`, `/feedback` endpoints with Pydantic schemas. |
| 3 | **Docker** | Multi-stage Dockerfile + `docker-compose.yml`. `docker compose up` works. |
| 4 | **pytest suite** | Unit + API tests. ≥80% coverage. `pytest --cov` runs clean. |
| 5 | **Structured logging** | JSON logs with `request_id`, `endpoint`, `latency_ms` on every request. |
| 6 | **README** | Setup in 3 commands. Architecture diagram. What the model predicts. |

## Homework tasks explicitly called out during the session
- [ ] Move any notebook-only code into proper `.py` scripts under `src/`
- [ ] Try `git clone`, `git push`, `git pull` if unfamiliar
- [ ] Read up on **`typing.Protocol`** (structural typing) — see [[02 - Python Packaging and Project Structure]] for why this repo uses it instead of `ABC`
- [ ] Try `docker login`, explore Docker Hub, tag and push an image
- [ ] Look into **integration testing** and **end-to-end testing** — see the testing pyramid in [[06 - Structured Logging and pytest]]
- [ ] Try writing a `pytest` unit test for every function in your own project

## Reminder: this is NOT the final course project
The **real, graded** deliverable (tied to the Project Certificate) is one of the two tracks — see [[MLOps Practitioner Course - Project Track Decision]] for the Track 1 (Deep Learning / Arabic Sentiment Analysis) vs Track 2 (LLM/RAG / Arabic Legal Q&A) decision, rubric, and timeline. Don't confuse the two — this session's exercise won't be reviewed by anyone; the track project will be.

> [!info] Timeline reminder
> - 1 GitHub commit **per session** (branch-per-session or branch-per-concept — not one giant commit at the end)
> - Peer review of a classmate's project required
> - Final submission: within 2 weeks after Session 5
