"""Load .env at the package boundary, so every entry point sees the same config.

`ollama_langfuse_rag.py` loads it at import; anything that does NOT go through
that module (evals/run_ragas.py --source langfuse, tools/cost_report.py, the
judge's push_score) would otherwise construct a Langfuse client with no keys and
fail with "initialized without public_key" — which reads like an auth problem
and is actually a working-directory problem.

Doing it here means it happens exactly once per package, before any module-level
client is built.
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # optional — export the variables yourself instead
    pass
