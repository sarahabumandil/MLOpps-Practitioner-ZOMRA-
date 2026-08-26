---
tags: [mlops, session1, docker]
up: "[[00 - MLOps S1 - From Code to Container]]"
---

# Topic 05 · Docker & Containerization

> [!danger] This note replaces an earlier version that had an invented Dockerfile
> My first pass at this note wrote a Dockerfile from memory of the lecture — wrong base image version, wrong copy strategy, wrong username command. Everything below is copied directly from the real `Dockerfile` and `docker-compose.yml` in the repo. If you study only one note critically before an exam/interview, make it this one, since Docker is described as the single most commonly interview-tested MLOps skill.

## The problem Docker solves
"It works on my machine" — different Python versions, different library versions, missing system libraries, different OS path separators between dev/CI/prod.

**Docker's answer**: one image, identical behavior everywhere it runs.

## Core vocabulary

| Concept | What it is | Analogy |
|---|---|---|
| **Dockerfile** | Text file of instructions (`FROM`, `COPY`, `RUN`, `CMD`) describing how to build the environment | The *recipe* |
| **Image** | The built, immutable snapshot — layers of filesystem + metadata, tagged (e.g. `ride-duration-api:session1`) | The *frozen meal* — cook once, reheat anywhere |
| **Container** | A *running instance* of an image — isolated process with its own filesystem/network/resources. Many containers can run from one image; changes inside a container die with it | The *served plate* — one recipe, many servings |
| **Registry** | A server that stores/distributes images, so any machine can `docker pull` and run your code without ever seeing the source | The *supermarket* for images |

Lifecycle:
```
Dockerfile ──docker build──▶ Image ──docker run──▶ Container
                              │
                              └──docker push/pull──▶ Registry
```
The default registry is **Docker Hub** (that's where `python:3.12-slim` comes from). Alternatives: `ghcr.io` (GitHub Container Registry — where this repo's MLflow image lives), AWS ECR, Google Artifact Registry. This repo's CI pushes its verified image to Docker Hub on every merge to `main`.

## The actual Dockerfile — full, unmodified
```dockerfile
# syntax=docker/dockerfile:1

# ── Stage 1: builder ──────────────────────────────────
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install .

# ── Stage 2: runtime ──────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY fastapi_example.py ./

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "fastapi_example:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Line-by-line, what each part is doing and why

| Line(s) | What it does | Why |
|---|---|---|
| `FROM python:3.12-slim AS builder` | Names this stage `builder` so a later stage can reference it | Multi-stage build — this stage's job is *only* to install dependencies |
| `ENV PIP_NO_CACHE_DIR=1` | Tells pip not to keep a local wheel cache | Smaller intermediate layer — no point caching wheels we'll discard |
| `RUN python -m venv /opt/venv` then `ENV PATH="/opt/venv/bin:$PATH"` | Creates a **virtualenv inside the container itself**, then puts it first on `PATH` | This is the key trick of this Dockerfile: dependencies get installed into an isolated venv folder, so the *entire* runtime environment can be copied as one clean unit into the next stage |
| `WORKDIR /build` | Sets working directory for this stage | Just a scratch location — not the same path as the runtime stage's `/app` |
| `COPY pyproject.toml README.md ./` then `COPY src ./src` | Copies only what `pip install .` needs, **before** installing | **Layer caching**: as long as these files don't change, Docker reuses the cached "install" layer on rebuilds — much faster iteration |
| `RUN pip install .` | Installs the project (fastapi, uvicorn, pydantic, litestar) into `/opt/venv` | Note: the `onnx` extra is deliberately **excluded** here — the served API doesn't need torch/onnx at runtime |
| `FROM python:3.12-slim AS runtime` | **Fresh** base image for the second stage | Nothing from the builder stage carries over automatically — that's the whole point |
| `ENV PYTHONUNBUFFERED=1` | Forces stdout/stderr to flush immediately instead of buffering | So logs/prints show up in real time in `docker logs`, not delayed |
| `ENV PYTHONDONTWRITEBYTECODE=1` | Stops Python writing `.pyc` files | No point — the container is ephemeral, and it avoids clutter |
| `COPY --from=builder /opt/venv /opt/venv` | Copies the **entire ready-made virtualenv** from the builder stage | This is *the* multi-stage payoff: the runtime image never had pip, build tools, or the pip cache — only the finished venv arrives |
| `WORKDIR /app` | Working directory for the actual app | Separate from the builder's `/build` |
| `COPY fastapi_example.py ./` | Copies just the entry-point script | `src/` is **not** copied again here — it's already installed as a package *inside* the venv that was just copied over |
| `RUN useradd --create-home --uid 1000 appuser` then `USER appuser` | Creates a non-root user and switches to it | If the app is ever exploited, the damage is contained — never run production containers as `root` |
| `EXPOSE 8000` | Documents which port the container listens on | Informational — doesn't actually publish the port (that's `-p` at `docker run` time) |
| `CMD [...]` | The default command when the container starts | Binds to `0.0.0.0`, **not** `127.0.0.1` — see note below |

> [!important] `0.0.0.0` vs `127.0.0.1` inside a container
> The comment in the actual Dockerfile says it directly: *"Bind to 0.0.0.0 so the app is reachable from outside the container."* If you bind to `127.0.0.1` inside a container, the app only accepts connections from *inside that same container* — nothing on the host machine can reach it, even with the right `-p` mapping. This is a very common first-time Docker mistake. Note the contrast: `docs_info` shows the *local, non-Docker* dev command using `127.0.0.1` — that's fine outside a container; it must change to `0.0.0.0` once containerized.

### Multi-stage build — why bother
- **Builder stage**: has pip, build tools, potentially compiler toolchains for anything with C extensions
- **Runtime stage**: has *only* the finished venv + the entry-point script + a non-root user
- Result: none of the build-time tooling ends up in the final image → smaller image, smaller attack surface

### Building and running this project's image
```bash
docker build -t ride-duration-api:session1 .
docker run --rm -p 8000:8000 ride-duration-api:session1
```
- `--rm` → auto-remove the container once it exits, so you don't accumulate dead containers
- `-p 8000:8000` → maps `host_port:container_port` — without this, the app runs *inside* the container but is unreachable from your browser
- API then available at `http://127.0.0.1:8000` — same endpoints/docs as running it un-containerized. Stop with `Ctrl+C`.

## `docker-compose.yml` — the actual file, in full
```yaml
services:

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/v1/model.pkl
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/models:ro  # read-only model mount
    depends_on:
      - mlflow
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.12.1
    ports:
      - "5000:5000"
    command: >
      mlflow server
        --backend-store-uri sqlite:///mlflow.db
        --default-artifact-root /artifacts
        --host 0.0.0.0
    volumes:
      - mlflow-data:/artifacts

volumes:
  mlflow-data:
```

> [!note] The obsolete `version:` key
> Older Compose tutorials show a top-level `version: "3.8"` line at the very top of the file. It's **obsolete in Compose v2** and deliberately omitted here — if you see it in older examples online, you don't need to add it.

| Service | Image / build | Port | Purpose |
|---|---|---|---|
| `api` | built from this repo's `Dockerfile` | `8000` | Ride Duration API |
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.12.1` | `5000` | Experiment tracking + artifact store |

| Field | What it does |
|---|---|
| `environment: MODEL_PATH=/models/v1/model.pkl` | Sets where the app *should* look for a real model — see the important caveat below |
| `volumes: ./models:/models:ro` | Mounts the local `models/` folder **read-only** into the container at `/models` |
| `depends_on: [mlflow]` | `api` waits for `mlflow` to start first — this is *start-order* only, not a readiness check (add health-check conditions for real robustness) |
| `restart: unless-stopped` | Auto-restarts the container if it crashes, unless you deliberately stopped it |

> [!danger] `MODEL_PATH` is set here, but nothing reads it yet
> Compose sets `MODEL_PATH` and mounts `models/` read-only, but as covered in [[03 - FastAPI for ML Inference]], **`fastapi_example.py` never actually reads that environment variable** — it always uses the built-in heuristic estimator. This is the #1 item on the README's "🔜 next steps" list: load the pickle from `MODEL_PATH` at startup (via FastAPI's *lifespan*), falling back to the heuristic with a loud log line if it's missing. Drop a real model file into `models/v1/model.pkl` today and the app still won't touch it — that wiring simply isn't built yet.

### Running the stack
```bash
docker compose up --build          # foreground
docker compose up -d --build       # detached (background)
docker compose down                # stop + remove containers
docker compose down -v             # also remove the mlflow-data volume
```
- API: `http://127.0.0.1:8000`
- MLflow UI: `http://127.0.0.1:5000`

## Quick reference

| Command | What it does |
|---|---|
| `docker build -t ride-duration-api:session1 .` | Build the image |
| `docker run --rm -p 8000:8000 ride-duration-api:session1` | Run it, auto-remove on exit |
| `docker compose up --build` | Build + start every service |
| `docker compose up -d --build` | Same, detached |
| `docker compose down` | Stop and remove containers |
| `docker compose down -v` | Also remove named volumes (e.g. `mlflow-data`) |
| `docker run -d -p 6379:6379 redis:7-alpine` | Start Redis for the MessagePack cache demo (see [[04 - Serialization Formats and ONNX]]) |

## Not yet in this repo (see [[09 - What This Session Deliberately Leaves Out]])
- A `HEALTHCHECK` instruction in the Dockerfile / compose, tied to a *real* health check that verifies the model is loaded
- A lockfile-pinned build (`uv lock` / `pip-compile`) — today's `>=` version bounds reproduce the *project*, not a byte-identical *environment*
