# 08. PostgreSQL Status Web App

This example runs a small Flask application alongside PostgreSQL. The web page and `/health` endpoint report whether the application can connect to the database.

## Project structure

```text
08-postgres-status-app/
├── README.md
├── compose.yaml
├── Dockerfile
├── .dockerignore
├── app.py
└── requirements.txt
```

## Run the stack

```bash
docker compose config
docker compose up --detach --build
docker compose ps
```

Open <http://localhost:5000>. The response includes the PostgreSQL connection status and server version. The same status is available as JSON from <http://localhost:5000/health>.

The application uses the Compose service name `db` as its database hostname. It does not use `localhost`, because `localhost` inside the app container refers to the app container itself.

## Useful commands

```bash
docker compose logs --follow app
docker compose exec db psql -U appuser -d appdb
docker compose down
docker compose down --volumes  # Also remove the database data.
```

## What to observe

- `depends_on` waits for the PostgreSQL healthcheck before starting the app.
- The named volume preserves database data across normal container recreation.
- The application returns HTTP 503 from `/health` when PostgreSQL is unavailable.
- Credentials are supplied through Compose environment variables for local learning; production deployments should use a secret manager.
