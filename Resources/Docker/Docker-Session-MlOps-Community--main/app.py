import os

import psycopg
from flask import Flask

app = Flask(__name__)


def database_status():
    try:
        with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=3) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]
        return {"status": "healthy", "version": version}
    except (KeyError, psycopg.Error) as error:
        app.logger.warning("PostgreSQL check failed: %s", error)
        return {"status": "unhealthy", "error": "database connection failed"}


@app.get("/")
def index():
    status = database_status()
    response = {"application": "healthy", "postgres": status}
    return response, 200 if status["status"] == "healthy" else 503


@app.get("/health")
def health():
    status = database_status()
    return status, 200 if status["status"] == "healthy" else 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
