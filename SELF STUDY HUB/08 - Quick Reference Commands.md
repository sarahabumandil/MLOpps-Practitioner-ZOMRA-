---
tags: [mlops, session1, cheatsheet]
up: "[[00 - MLOps S1 - From Code to Container]]"
---

# Topic 08 · Quick Reference — All Commands in One Place

## `uv` (Python package manager)
```bash
uv init my_project     # scaffold new project
uv add fastapi          # add + install a dependency
uv sync                  # install everything from pyproject.toml + lockfile
```

## Docker
| Command | What it does |
|---|---|
| `docker build -t my-api .` | Build image from Dockerfile here |
| `docker run -p 8000:8000 my-api` | Run, mapping host:8000 → container:8000 |
| `docker run -d --name api my-api` | Run detached, with a name |
| `docker ps` | List running containers |
| `docker ps -a` | List ALL containers (incl. stopped) |
| `docker exec -it api /bin/bash` | Shell into a running container |
| `docker logs -f api` | Tail live logs |
| `docker images` | List local images |
| `docker rmi my-api` | Delete an image |
| `docker stop api && docker rm api` | Stop and remove a container |
| `docker login` | Authenticate with Docker Hub |
| `docker tag test mlops-session1:1.0` | Tag an image |
| `docker push <user>/mlops-session1:1.0` | Push image to registry |
| `docker compose up --build` | Build + start every service (foreground) |
| `docker compose up -d --build` | Same, detached |
| `docker compose ps` | List running compose services |
| `docker compose logs -f api` | Tail logs for one service |
| `docker compose down` | Stop and remove containers |
| `docker compose down -v` | Also remove named volumes (e.g. `mlflow-data`) |

## pytest
| Command | What it does |
|---|---|
| `pytest` | Run all tests here |
| `pytest tests/test_model.py` | Run one file |
| `pytest tests/test_model.py::test_predict` | Run one test |
| `pytest -k "predict"` | Run tests whose name contains "predict" |
| `pytest -m slow` | Run tests marked `@pytest.mark.slow` |
| `pytest -v` | Verbose |
| `pytest --cov=src` | Run with coverage |
| `pytest --cov=src --cov-report=html` | HTML report → `htmlcov/` |
| `pytest --cov=src --cov-fail-under=80` | Fail CI below 80% |
| `pytest -x` | Stop at first failure |
| `pytest -s` | Show `print()` output |
| `pytest --tb=short` | Short traceback |

## FastAPI / uvicorn (real entry points from this repo)
```bash
python fastapi_example.py                                       # simplest, runs __main__ block
uvicorn fastapi_example:app --host 127.0.0.1 --port 8000 --reload   # dev, with auto-reload

# Litestar equivalent:
python litestar_example.py
litestar --app litestar_example:app run --host 127.0.0.1 --port 8001 --reload
```
→ FastAPI docs: `http://127.0.0.1:8000/docs` · Litestar docs: `http://127.0.0.1:8001/schema/swagger`

> [!warning] Inside a Docker container, the host changes
> `--host 127.0.0.1` is correct for local dev, but **must be `0.0.0.0` inside a container** — see [[05 - Docker and Containerization]] for why.

## Git (mentioned as prerequisite / homework)
```bash
git clone <repo-url>
git pull
git push
```

## Redis (for the MessagePack caching demo)
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

## 📚 Resources
> [!warning] Unverified section
> I don't have a confirmed source (slide, README, or transcript) for this specific list of links — flagging that honestly rather than presenting it as verified. Treat these as general-purpose recommendations for the tools this session covers, not as a transcribed list of what was specifically shared. Cross-check before citing any of these as "Aya said."

| Resource | Link | Note |
|---|---|---|
| FastAPI docs | fastapi.tiangolo.com/tutorial | Official tutorial |
| Litestar docs | docs.litestar.dev | Official docs — covers the DI pattern used in `litestar_example.py` |
| pytest docs | docs.pytest.org | Fixtures, parametrize, marks |
| uv docs | docs.astral.sh/uv | Official docs for the package manager used throughout |
| ONNX docs | onnx.ai | Official docs, including the opset/operator catalog |
| Docker docs | docs.docker.com/get-started | Official Getting Started series |

## 🔮 Coming in Session 2 (Aug 23) — per your memory notes, not verified against session 2 source
The Docker image built today becomes the input to Session 2's automated pipeline: MLflow, DVC, CI/CD, Terraform. I don't have Session 2's actual materials yet, so treat this line as a placeholder until that session's real content is available to check against.

> [!todo] Before Session 2
> - Push your repo
> - Confirm `docker compose up --build` runs cleanly
> - Make sure `pytest` passes (expect `5 passed`)
