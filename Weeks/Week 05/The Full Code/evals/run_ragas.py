"""RAGAS on Ollama, in three modes, with the reporting that makes it honest.

    python -m evals.run_ragas --source dataset --gate --limit 10
    python -m evals.run_ragas --source langfuse --hours 24 --sample 0.03 --push
    python -m evals.run_ragas --source dataset --compare baseline

Run it on the same ten samples as `evals/judge.py` and compare. Where the two
agree the number is probably real; where they diverge is the calibration lesson,
because they are two different rulers measuring the same idea.

Three things this file refuses to do, each because doing them is how eval
harnesses end up reporting numbers nobody should act on:

* **Print a bare mean.** A bimodal 0.85 — half the answers perfect, half broken
  — is a completely different system from a uniform 0.85, and the mean cannot
  tell them apart. The distribution and the worst items are printed too.
* **Ignore NaNs.** When the judge's output will not parse, RAGAS retries and
  then drops the sample. The dropped samples are the HARD ones, so the surviving
  mean looks better than reality. Above a 10% NaN rate the run fails.
* **Gate on the global mean alone.** A healthy average with one collapsed
  language is exactly how incident 09 stayed invisible for four hours. Any
  single language falling more than 0.10 below the threshold fails the run, even
  when the overall number passes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

RUNS_DIR = Path(__file__).resolve().parent / "runs"

#: Same tag as the judge, so exactly one model stays resident in VRAM.
MODEL = os.getenv("OLLAMA_JUDGE_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
SEED = 42

#: Match OLLAMA_NUM_PARALLEL. Every concurrent request gets its own slice of the
#: KV cache, so raising this while num_ctx is 8192 does not make the run faster
#: — it makes long Arabic contexts evict each other, and on a 24 GB card it
#: stalls or OOMs. Parallelism here is a cliff, not a slope.
MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", os.getenv("OLLAMA_NUM_PARALLEL", "4")))

#: Above this share of unscored samples the mean is computed over a biased,
#: easier subset. A metric that quietly scored 40 of 60 items is not a measurement.
NAN_LIMIT = 0.10
#: A single language may not fall more than this below the global threshold.
SEGMENT_TOLERANCE = 0.10


def embeddings_available() -> bool:
    """Probe whether this Ollama server was started with embeddings enabled.

    `ollama serve` without `--embeddings` answers 501 to /api/embed, and the
    only RAGAS metric that needs them (ResponseRelevancy) then fails every
    sample. Better to say so once, up front, than to report a column of NaNs.
    """
    try:
        resp = httpx.post(
            f"{BASE_URL}/api/embed", json={"model": MODEL, "input": "probe"}, timeout=30.0
        )
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def build_metrics(with_embeddings: bool) -> tuple[list, list]:
    """Wire RAGAS to Ollama.

    Returns (metrics, reference_metrics). They are separated because RAGAS
    validates required columns across the WHOLE dataset: one adversarial item
    with no ground truth would make context recall reject the entire run. So
    the reference-only metrics get their own pass over the referenced subset,
    which is also what "only when the sample carries a reference" means.
    """
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        ContextRecall,
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )

    llm = LangchainLLMWrapper(
        ChatOllama(
            model=MODEL,
            base_url=BASE_URL,
            temperature=0,
            seed=SEED,
            # Explicit, always. Ollama's 2048 default truncates the answer plus
            # every retrieved chunk, and Arabic reaches the ceiling first — so
            # you would be measuring your own truncation, in one language only.
            num_ctx=NUM_CTX,
            format="json" if os.getenv("RAGAS_FORMAT_JSON", "0") == "1" else None,
        )
    )
    metrics: list = [Faithfulness(llm=llm), LLMContextPrecisionWithoutReference(llm=llm)]
    if with_embeddings:
        embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", MODEL), base_url=BASE_URL)
        )
        metrics.append(ResponseRelevancy(llm=llm, embeddings=embeddings))
    # Needs ground truth, so it can never run on live production traffic at all.
    return metrics, [ContextRecall(llm=llm)]


# ══════════════════════════════════════════════════════════════════════
#  Sources
# ══════════════════════════════════════════════════════════════════════


def from_dataset(limit: int = 0) -> list[dict[str, Any]]:
    """Run the RAG over the frozen testset and collect what RAGAS needs."""
    import ollama_langfuse_rag as rag
    from evals.build_testset import load

    items = load()
    if limit:
        # Stratify by (language, source). Slicing the file instead would take the
        # adversarial block for one language and the synthetic block for the
        # other, and the resulting per-language gap would be an artifact of the
        # sampling rather than a property of the system — which is exactly the
        # mistake this file spends the rest of its lines guarding against.
        buckets: dict[tuple[str, str], list] = {}
        for item in items:
            buckets.setdefault((item["lang"], item["source"].split(":")[0]), []).append(item)
        picked, i = [], 0
        while len(picked) < limit and any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key] and len(picked) < limit:
                    picked.append(buckets[key].pop(0))
            i += 1
            if i > limit:
                break
        items = picked[:limit]

    samples = []
    for item in items:
        result = rag.rag_pipeline(item["question"])
        samples.append(
            {
                "id": item["id"],
                "lang": item["lang"],
                "user_input": item["question"],
                "response": result["answer"],
                "retrieved_contexts": [d["text"] for d in result["docs"]] or [""],
                "reference": item.get("reference") or None,
                "expects_refusal": item.get("expects_refusal", False),
            }
        )
    return samples


def _question_from(trace_input: Any) -> str:
    """Pull the question out of a trace's recorded input.

    `@observe` records a function's arguments, so a trace's input is
    `{"args": ["the question"], "kwargs": {}}` rather than the string itself.
    Assuming a bare string here silently yields an empty question for every
    trace, and the run then reports "no samples to score" as though production
    had been quiet — which is a very convincing way to measure nothing.
    """
    if isinstance(trace_input, str):
        return trace_input
    if isinstance(trace_input, dict):
        args = trace_input.get("args") or []
        if args and isinstance(args[0], str):
            return args[0]
        for key in ("question", "query", "input", "user_input"):
            value = trace_input.get(key)
            if isinstance(value, str):
                return value
    return ""


def from_langfuse(hours: int, sample_rate: float) -> list[dict[str, Any]]:
    """Pull yesterday's traces and score a sample of them.

    The question, the retrieved contexts and the answer are already ON the
    trace — that is the payoff for instrumenting retrieval as its own span in
    ollama_langfuse_rag.py. Nothing has to be re-run to evaluate production.
    """
    import random

    from langfuse import get_client

    client = get_client()
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    traces = client.api.trace.list(from_timestamp=since, limit=100).data
    rng = random.Random(SEED)
    picked = [t for t in traces if rng.random() < sample_rate]

    samples = []
    for trace in picked:
        full = client.api.trace.get(trace.id)
        contexts, response = [], ""
        question = _question_from(trace.input)
        for obs in getattr(full, "observations", []) or []:
            if obs.name == "retrieve" and isinstance(obs.output, list):
                contexts = [c.get("text", "") for c in obs.output if isinstance(c, dict)]
            if obs.type == "GENERATION" and obs.output:
                response = obs.output if isinstance(obs.output, str) else str(obs.output)
        if not (question and response):
            continue
        tags = getattr(trace, "tags", []) or []
        lang = next((t.split(":", 1)[1] for t in tags if t.startswith("lang:")), "en")
        samples.append(
            {
                "id": trace.id,
                "trace_id": trace.id,
                "lang": lang,
                "user_input": question,
                "response": response,
                "retrieved_contexts": contexts or [""],
                "reference": None,
                "expects_refusal": False,
            }
        )
    return samples


# ══════════════════════════════════════════════════════════════════════
#  Scoring and reporting
# ══════════════════════════════════════════════════════════════════════


def score(
    samples: list[dict[str, Any]], metrics: list, reference_metrics: list | None = None
) -> dict[str, dict[str, list]]:
    """Score all samples, then the referenced subset, and merge the results."""
    grouped = _score_subset(samples, metrics)
    referenced = [s for s in samples if s.get("reference")]
    if reference_metrics and referenced:
        print(
            f"  + {len(referenced)}/{len(samples)} samples carry a reference: "
            f"also scoring {', '.join(m.name for m in reference_metrics)}"
        )
        grouped.update(_score_subset(referenced, reference_metrics))
    return grouped


def _score_subset(samples: list[dict[str, Any]], metrics: list) -> dict[str, dict[str, list]]:
    """Run RAGAS over one set of samples and regroup as metric -> lang -> rows."""
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig

    dataset = EvaluationDataset.from_list(
        [
            {
                k: v
                for k, v in s.items()
                if k in {"user_input", "response", "retrieved_contexts", "reference"}
                and v is not None
            }
            for s in samples
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(max_workers=MAX_WORKERS, timeout=180, max_retries=3),
        show_progress=True,
    )
    frame = result.to_pandas()

    grouped: dict[str, dict[str, list]] = {}
    for metric in [m.name for m in metrics]:
        if metric not in frame.columns:
            continue
        grouped[metric] = {}
        for i, sample in enumerate(samples):
            grouped[metric].setdefault(sample["lang"], []).append(
                {
                    "id": sample["id"],
                    "value": float(frame[metric].iloc[i]),
                    "trace_id": sample.get("trace_id"),
                    "question": sample["user_input"],
                    "expects_refusal": sample.get("expects_refusal", False),
                }
            )
    return grouped


def histogram(values: list[float], width: int = 24) -> str:
    """Ten-bucket distribution, because a mean cannot show bimodality."""
    buckets = [0] * 10
    for v in values:
        buckets[min(int(v * 10), 9)] += 1
    peak = max(buckets) or 1
    lines = []
    for i, count in enumerate(buckets):
        bar = "#" * int(count / peak * width)
        lines.append(f"        {i / 10:.1f}-{(i + 1) / 10:.1f}  {count:>3} {bar}")
    return "\n".join(lines)


def report(grouped: dict[str, dict[str, list]], threshold: float) -> bool:
    """Print per-language means, NaN rates, distribution and worst items."""
    print(f"\n  judge: {MODEL} @ num_ctx={NUM_CTX} seed={SEED} workers={MAX_WORKERS}")
    passed = True

    for metric, by_lang in grouped.items():
        print(f"\n  {metric}")
        all_values: list[float] = []
        lang_means: dict[str, float] = {}
        n_refusals = sum(1 for rows in by_lang.values() for r in rows if r.get("expects_refusal"))

        for lang, rows in sorted(by_lang.items()):
            # Items whose correct answer is "I don't know" are excluded from the
            # gate. Faithfulness over a refusal is meaningless — there are no
            # claims to ground — and leaving them in means a testset that
            # deliberately contains unanswerable questions can never pass its own
            # gate. They are counted and reported, never silently dropped.
            rows = [r for r in rows if not r.get("expects_refusal")]
            values = [r["value"] for r in rows if not math.isnan(r["value"])]
            n_nan = len(rows) - len(values)
            nan_rate = n_nan / len(rows) if rows else 0.0
            mean = statistics.mean(values) if values else float("nan")
            lang_means[lang] = mean
            all_values += values
            flag = ""
            if nan_rate > NAN_LIMIT:
                flag, passed = "  <-- NaN RATE TOO HIGH", False
            print(
                f"    {lang}:  mean {mean:5.3f}  n={len(values):<4} "
                f"NaN {n_nan}/{len(rows)} = {nan_rate:4.0%}{flag}"
            )

        if n_refusals:
            refusal_rows = [
                r
                for rows in by_lang.values()
                for r in rows
                if r.get("expects_refusal") and not math.isnan(r["value"])
            ]
            confabulated = [r for r in refusal_rows if r["value"] > 0.5]
            print(
                f"    refusals: {n_refusals} unanswerable items excluded from the gate — "
                f"{len(confabulated)} scored >0.5, i.e. the model answered anyway"
            )

        if all_values:
            print(f"    all: mean {statistics.mean(all_values):5.3f}   distribution:")
            print(histogram(all_values))

        worst = sorted(
            (
                r
                for rows in by_lang.values()
                for r in rows
                if not math.isnan(r["value"]) and not r.get("expects_refusal")
            ),
            key=lambda r: r["value"],
        )[:5]
        if worst:
            print("    worst 5:")
            for r in worst:
                print(f"      {r['value']:5.3f}  {r['id']:<10} {r['question'][:48]}")

        # ── the segment gate ──────────────────────────────────────────
        for lang, mean in lang_means.items():
            if mean == mean and mean < threshold - SEGMENT_TOLERANCE:
                print(
                    f"    SEGMENT GATE FAILED: {lang} at {mean:.3f} is more than "
                    f"{SEGMENT_TOLERANCE} below the {threshold} threshold"
                )
                passed = False
        if all_values and statistics.mean(all_values) < threshold:
            print(f"    GATE FAILED: mean below {threshold}")
            passed = False
    return passed


def push_scores(grouped: dict[str, dict[str, list]]) -> int:
    """Write scores back onto the original traces, idempotently."""
    from langfuse import get_client

    client = get_client()
    pushed = 0
    for metric, by_lang in grouped.items():
        for rows in by_lang.values():
            for row in rows:
                if not row.get("trace_id") or math.isnan(row["value"]):
                    continue
                client.create_score(
                    name=f"ragas_{metric}",
                    value=row["value"],
                    trace_id=row["trace_id"],
                    score_id=f"{row['trace_id']}-ragas_{metric}",
                    data_type="NUMERIC",
                    comment=f"{MODEL} @ num_ctx={NUM_CTX}",
                )
                pushed += 1
    client.flush()
    return pushed


def save_run(name: str, grouped: dict[str, dict[str, list]]) -> Path:
    """Persist a run so a later one can diff against it per item."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{name}.json"
    path.write_text(json.dumps(grouped, indent=2, ensure_ascii=False))
    return path


def compare(grouped: dict[str, dict[str, list]], baseline_name: str) -> None:
    """Per-item diff against a saved run — surface what REGRESSED.

    A mean that moved from 0.81 to 0.83 can still hide six items that got
    materially worse. Promotion decisions are made on the regressions, not on
    the average.
    """
    path = RUNS_DIR / f"{baseline_name}.json"
    if not path.exists():
        print(f"  no baseline run '{baseline_name}' in {RUNS_DIR}/ — save one first")
        return
    baseline = json.loads(path.read_text())
    print(f"\n  per-item diff against '{baseline_name}':")
    for metric, by_lang in grouped.items():
        base_rows = {
            r["id"]: r["value"] for rows in baseline.get(metric, {}).values() for r in rows
        }
        deltas = []
        for rows in by_lang.values():
            for r in rows:
                if r["id"] in base_rows and not math.isnan(r["value"]):
                    deltas.append((r["value"] - base_rows[r["id"]], r["id"], r["question"]))
        regressions = sorted(d for d in deltas if d[0] < -0.05)
        print(f"    {metric}: {len(regressions)} regressed of {len(deltas)} compared")
        for delta, item_id, question in regressions[:5]:
            print(f"      {delta:+.3f}  {item_id:<10} {question[:48]}")


def main() -> None:
    """CLI entry point — see the module docstring for the three modes."""
    parser = argparse.ArgumentParser(prog="python -m evals.run_ragas", description=__doc__)
    parser.add_argument("--source", choices=("dataset", "langfuse"), default="dataset")
    parser.add_argument("--gate", action="store_true", help="exit non-zero below threshold")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--limit", type=int, default=0, help="score only N items")
    parser.add_argument("--hours", type=int, default=24, help="langfuse mode: lookback")
    parser.add_argument("--sample", type=float, default=0.03, help="langfuse mode: share to score")
    parser.add_argument("--push", action="store_true", help="write scores back to Langfuse")
    parser.add_argument("--save", metavar="NAME", help="persist this run for later comparison")
    parser.add_argument("--compare", metavar="NAME", help="per-item diff against a saved run")
    args = parser.parse_args()

    samples = (
        from_dataset(args.limit)
        if args.source == "dataset"
        else from_langfuse(args.hours, args.sample)
    )
    if not samples:
        raise SystemExit("no samples to score")

    with_embeddings = embeddings_available()
    if not with_embeddings:
        print(
            "\n  NOTE: this Ollama server has embeddings disabled (POST /api/embed -> 501),\n"
            "  so ResponseRelevancy is SKIPPED — it is the only metric here that needs them.\n"
            "  Faithfulness and context precision are LLM-only and run normally.\n"
            "  Fix: restart with `ollama serve --embeddings`, or set OLLAMA_EMBED_MODEL to a\n"
            "  pulled embedding model such as nomic-embed-text.\n"
        )
    metrics, reference_metrics = build_metrics(with_embeddings)
    print(f"  scoring {len(samples)} samples with: {', '.join(m.name for m in metrics)}")

    grouped = score(samples, metrics, reference_metrics)
    passed = report(grouped, args.threshold)

    if args.push:
        print(f"\n  pushed {push_scores(grouped)} scores back onto their traces")
    if args.save:
        print(f"  saved run -> {save_run(args.save, grouped)}")
    if args.compare:
        compare(grouped, args.compare)

    sys.exit(0 if passed or not args.gate else 1)


if __name__ == "__main__":
    main()
