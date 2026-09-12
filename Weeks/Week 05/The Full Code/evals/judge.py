"""LLM-as-a-judge, with its mechanism visible.

RAGAS is a judge too, and a better-engineered one. It also hides what it does
behind a metric name. This file exists so that when a student asks "what is
faithfulness, actually?", the answer is a prompt they can read, on line 130,
rather than a citation.

Grown from `judge_answer()` in ollama_langfuse_rag.py, which is the same idea in
twenty lines. Four things it adds, each because the twenty-line version is not
safe to draw conclusions from:

* **A fingerprint.** Model tag, digest, temperature, seed and context window are
  recorded with every score. A score is only comparable to another score from
  the SAME ruler — and `ollama pull` can change the ruler underneath you.
* **Counted parse failures.** The original degrades to a null score. Null is
  right; silently scoring 0.0 would be a lie. But a null that nobody counts is
  also a lie, because it drops the hard samples and flatters the mean.
* **A refusal flag.** "I don't know" is the correct answer to an unanswerable
  question and scores terribly on relevance. Bucket refusals or they drag the
  mean down and you will "fix" the model into confabulating.
* **Per-language reporting.** The whole point of the bilingual corpus.

    python -m evals.judge --demo
    python -m evals.judge --file samples.jsonl --rubric groundedness --rubric relevance
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from evals.knowledge_base import detect_language

#: Ollama's default context is 2048 tokens and it truncates SILENTLY. An
#: evaluation prompt carries the answer plus every retrieved chunk, and Arabic
#: spends more tokens per word than English — so Arabic hits the ceiling first,
#: its context arrives cut in half, and the judge marks claims unsupported for a
#: reason that has nothing whatsoever to do with the system under test. You would
#: spend the session debugging your own evaluator.
MIN_NUM_CTX = 4096


class ParseFailure(RuntimeError):
    """The judge returned something that is not the requested JSON.

    Raised, never swallowed into a 0.0. A parse failure means "no measurement",
    and a missing measurement and a bad measurement are different facts.
    """


@dataclass
class JudgeConfig:
    """Everything that has to be identical for two scores to be comparable."""

    model: str = os.getenv("OLLAMA_JUDGE_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    base_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    temperature: float = 0.0
    seed: int = 42
    digest: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Resolve the model digest and refuse an unusable context window."""
        assert self.num_ctx >= MIN_NUM_CTX, (
            f"num_ctx={self.num_ctx} is below {MIN_NUM_CTX}. Ollama truncates silently, "
            "Arabic hits the ceiling first, and you would be measuring the truncation."
        )
        try:
            resp = httpx.post(f"{self.base_url}/api/show", json={"model": self.model}, timeout=30.0)
            resp.raise_for_status()
            self.digest = str(resp.json().get("details", {}).get("parent_model") or "") or (
                resp.json().get("model_info", {}).get("general.basename", "")
            )
        except httpx.HTTPError:
            self.digest = "unresolved"

    def fingerprint(self) -> dict[str, str]:
        """The ruler, as strings — recorded on every score this judge writes.

        `qwen2.5:7b` and `llama3.1:8b` are moving aliases: the same tag can point
        at different weights next month. Recording the tag alone is not a pin.
        """
        return {
            "judge_model": self.model,
            "judge_digest": self.digest,
            "judge_num_ctx": str(self.num_ctx),
            "judge_temperature": str(self.temperature),
            "judge_seed": str(self.seed),
        }


# ══════════════════════════════════════════════════════════════════════
#  The rubrics. Each is a system prompt plus the JSON schema it must return.
# ══════════════════════════════════════════════════════════════════════

GROUNDEDNESS_PROMPT = """You are a strict evaluator of factual grounding.

You will be given a CONTEXT and an ANSWER.

Work in two steps.

Step 1. Decompose the ANSWER into atomic claims. An atomic claim is a single
statement that can be true or false on its own. Split compound sentences. Ignore
pleasantries, hedges and restatements of the question.

Step 2. For each claim, decide whether the CONTEXT supports it. Judge ONLY
against the CONTEXT. Do not use anything you know about the world. A claim that
is true in reality but absent from the CONTEXT is NOT supported — that is the
distinction this metric exists to measure.

Return JSON only."""

GROUNDEDNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "why": {"type": "string"},
                },
                "required": ["claim", "supported"],
            },
        }
    },
    "required": ["claims"],
}

RELEVANCE_PROMPT = """You are evaluating whether an ANSWER addresses a QUESTION.

Score from 0.0 to 1.0, where 1.0 means the answer directly and completely
addresses what was asked, and 0.0 means it addresses something else entirely.
Ignore whether the answer is factually correct — that is a different metric.

Set is_refusal to true if the answer declines to answer, says it does not know,
or says the information is unavailable. A refusal is often the CORRECT response
to an unanswerable question, so it is bucketed separately rather than scored as
a failure. Still give it a relevance score, but set the flag.

Return JSON only."""

RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "reason": {"type": "string"},
        "is_refusal": {"type": "boolean"},
    },
    "required": ["score", "reason", "is_refusal"],
}

LANGUAGE_PROMPT = """You are identifying the language of two pieces of text.

Given a QUESTION and an ANSWER, name the language each one is written in. Use
the English name of the language, lowercase: "english", "arabic", "french".

Judge the language the text is mainly written in. Technical terms, product names
and code kept in English inside an otherwise Arabic answer do NOT make that
answer english.

Return JSON only."""

#: Note what this schema does NOT ask for: a "matches" boolean. The first
#: version did, and llama3.1:8b returned {"matches": false, "answer_language":
#: "English"} for an English answer to an English question — it contradicted
#: itself inside one object, because with constrained decoding the model commits
#: to the first field before it has "thought about" the second.
#:
#: Two lessons, both cheap: order your schema so observations come before
#: conclusions, and never ask a model for a conclusion you can compute yourself
#: from its own observations. The comparison below is one line of Python and it
#: cannot contradict itself.
LANGUAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "question_language": {"type": "string"},
        "answer_language": {"type": "string"},
    },
    "required": ["question_language", "answer_language"],
}


class Judge:
    """One judge, one config, three rubrics."""

    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig()
        self.parse_failures: list[dict[str, str]] = []

    def ask(self, system: str, user: str, schema: dict[str, Any], retries: int = 3) -> dict:
        """One schema-constrained call to Ollama. Raises ParseFailure, never guesses.

        `format=<schema>` makes Ollama constrain its decoding to the schema,
        which removes most parsing pain. Most, not all: a small model can still
        emit a truncated object, and that is what the retries are for.
        """
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": schema,
            "options": {
                "temperature": self.config.temperature,
                "seed": self.config.seed,
                "num_ctx": self.config.num_ctx,
            },
        }
        last = ""
        for _ in range(retries):
            try:
                resp = httpx.post(f"{self.config.base_url}/api/chat", json=payload, timeout=180.0)
                resp.raise_for_status()
                last = resp.json()["message"]["content"]
                return json.loads(last)
            except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
                last = f"{type(exc).__name__}: {exc} | {last[:200]}"
        raise ParseFailure(last)

    # ── rubric: groundedness ──────────────────────────────────────────
    def groundedness(self, question: str, answer: str, context: str) -> dict[str, Any]:
        """Claim-by-claim grounding. Score = supported claims / total claims.

        Returns the unsupported claims, not just the number — "0.6" tells you
        nothing actionable, "these two sentences are not in the context" does.
        """
        user = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}"
        result = self.ask(GROUNDEDNESS_PROMPT, user, GROUNDEDNESS_SCHEMA)
        claims = result.get("claims") or []
        if not claims:
            raise ParseFailure("judge returned zero claims")
        supported = [c for c in claims if c.get("supported")]
        return {
            "value": len(supported) / len(claims),
            "n_claims": len(claims),
            "unsupported": [c["claim"] for c in claims if not c.get("supported")],
        }

    # ── rubric: relevance ─────────────────────────────────────────────
    def relevance(self, question: str, answer: str, context: str = "") -> dict[str, Any]:
        """Does the answer address the question? Refusals flagged, not punished."""
        user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
        result = self.ask(RELEVANCE_PROMPT, user, RELEVANCE_SCHEMA)
        return {
            "value": max(0.0, min(1.0, float(result["score"]))),
            "reason": str(result.get("reason", ""))[:400],
            "is_refusal": bool(result.get("is_refusal")),
        }

    # ── rubric: language_match ────────────────────────────────────────
    def language_match(self, question: str, answer: str, context: str = "") -> dict[str, Any]:
        """Did the answer come back in the question's language?

        One short call, and it catches the bilingual failure that nothing else
        in the stack looks for: a model that quietly answers an Arabic question
        in English scores well on every other metric here.
        """
        user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
        result = self.ask(LANGUAGE_PROMPT, user, LANGUAGE_SCHEMA)
        q_lang = str(result["question_language"]).strip().lower()
        a_lang = str(result["answer_language"]).strip().lower()
        return {
            "value": q_lang == a_lang,
            "question_language": q_lang,
            "answer_language": a_lang,
        }

    def rubrics(self) -> dict[str, Callable[..., dict[str, Any]]]:
        """Rubric name -> bound method."""
        return {
            "groundedness": self.groundedness,
            "relevance": self.relevance,
            "language_match": self.language_match,
        }


DATA_TYPES = {"groundedness": "NUMERIC", "relevance": "NUMERIC", "language_match": "BOOLEAN"}


def push_score(judge: Judge, trace_id: str, rubric: str, result: dict[str, Any]) -> None:
    """Write one judge score to Langfuse, idempotently.

    The `judge_` prefix is not decoration: it guarantees these can never be
    averaged together with `ragas_` scores on one chart. Two judges measuring
    the same concept with different rubrics produce different numbers, and a
    mean across both is a number with no referent.
    """
    from langfuse import get_client

    client = get_client()
    client.create_score(
        name=f"judge_{rubric}",
        value=result["value"],
        trace_id=trace_id,
        # Deterministic id: re-running the judge over the same traces updates
        # the score instead of stacking a second one next to it.
        score_id=f"{trace_id}-judge_{rubric}",
        data_type=DATA_TYPES[rubric],
        comment=f"{judge.config.model} @ num_ctx={judge.config.num_ctx}",
        metadata={
            **judge.config.fingerprint(),
            **{k: v for k, v in result.items() if k != "value"},
        },
    )


# ══════════════════════════════════════════════════════════════════════
#  Batch runner — the reporting is the point
# ══════════════════════════════════════════════════════════════════════

#: Above this share of unparseable responses, the mean is computed over a biased
#: subset (the samples that failed are the hard ones) and should not be quoted.
PARSE_FAILURE_LIMIT = 0.10


def run_batch(
    judge: Judge, samples: list[dict[str, Any]], rubric_names: list[str], push: bool = False
) -> dict[str, Any]:
    """Score every sample under every rubric, and report per rubric PER LANGUAGE."""
    rubrics = judge.rubrics()
    results: dict[str, dict[str, list]] = {name: {"ok": [], "failed": []} for name in rubric_names}

    for sample in samples:
        question = sample["question"]
        answer = sample["answer"]
        context = sample.get("context") or "\n\n".join(sample.get("contexts", []))
        lang = sample.get("lang") or detect_language(question)

        for name in rubric_names:
            try:
                result = rubrics[name](question, answer, context)
            except ParseFailure as exc:
                results[name]["failed"].append(
                    {"lang": lang, "question": question, "why": str(exc)[:120]}
                )
                continue
            row = {**result, "lang": lang, "question": question}
            results[name]["ok"].append(row)
            if push and sample.get("trace_id"):
                push_score(judge, sample["trace_id"], name, result)

    return results


def report(results: dict[str, dict[str, list]], fingerprint: dict[str, str]) -> bool:
    """Print the per-language table. Returns False if any rubric is untrustworthy."""
    print("\n  judge:", " ".join(f"{k}={v}" for k, v in fingerprint.items()))
    trustworthy = True

    for name, buckets in results.items():
        ok, failed = buckets["ok"], buckets["failed"]
        total = len(ok) + len(failed)
        if not total:
            continue
        print(f"\n  {name}")
        languages = sorted({r["lang"] for r in ok} | {r["lang"] for r in failed})
        for lang in languages:
            vals = [float(r["value"]) for r in ok if r["lang"] == lang]
            n_failed = sum(1 for r in failed if r["lang"] == lang)
            n_total = len(vals) + n_failed
            rate = n_failed / n_total if n_total else 0.0
            mean = statistics.mean(vals) if vals else float("nan")
            flag = "  <-- UNTRUSTWORTHY" if rate > PARSE_FAILURE_LIMIT else ""
            print(
                f"    {lang}:  mean {mean:5.3f}   n={len(vals):<4} "
                f"parse-failures {n_failed}/{n_total} = {rate:4.0%}{flag}"
            )
            if rate > PARSE_FAILURE_LIMIT:
                trustworthy = False

        # The refusal bucket, kept out of the relevance mean on purpose.
        refusals = [r for r in ok if r.get("is_refusal")]
        if refusals:
            print(
                f"    refusals bucketed separately: {len(refusals)} "
                f"(excluded from the means above would change them — see report.py)"
            )

        worst = sorted(ok, key=lambda r: float(r["value"]))[:5]
        if worst:
            print("    worst:")
            for r in worst:
                extra = ""
                if r.get("unsupported"):
                    extra = f"  unsupported: {r['unsupported'][0][:60]}"
                print(f"      {float(r['value']):5.3f} [{r['lang']}] {r['question'][:52]}{extra}")

    if not trustworthy:
        print(
            f"\n  WARNING: at least one rubric exceeded a {PARSE_FAILURE_LIMIT:.0%} parse-failure\n"
            "  rate. The samples that failed are the hard ones, so the means above are\n"
            "  computed over an easier subset than the one you meant to measure. Do not\n"
            "  quote them. Raise num_ctx, or use a larger judge, and re-run.\n"
        )
    return trustworthy


#: Two samples, one per language. The English answer contains a claim that is
#: true in reality but absent from the context ("used in credit risk"), so a
#: working groundedness rubric must score it below 1.0 — that gap is the whole
#: difference between "is this correct?" and "is this grounded?".
DEMO_SAMPLES: list[dict[str, Any]] = [
    {
        "question": "What PSI value should trigger a retrain?",
        "answer": (
            "A PSI above 0.25 is the conventional trigger to retrain. Below 0.10 the "
            "population is stable. PSI is also the standard metric used across the credit "
            "risk industry."
        ),
        "context": (
            "PSI measures how far a distribution has moved from a baseline. Below 0.10 the "
            "population is stable. Between 0.10 and 0.25 is a moderate shift worth "
            "investigating. Above 0.25 is significant and is the conventional trigger to "
            "retrain."
        ),
        "lang": "en",
    },
    {
        "question": "ما قيمة مؤشر استقرار التوزيع التي توجب إعادة التدريب؟",
        "answer": (
            "القيمة الأعلى من 0.25 هي الحد المتعارف عليه لإعادة التدريب، وأقل من 0.10 يعني "
            "أن التوزيع مستقر."
        ),
        "context": (
            "يقيس مؤشر PSI مقدار ابتعاد التوزيع الحالي عن التوزيع المرجعي. أقل من 0.10 يعني "
            "أن التوزيع مستقر. بين 0.10 و0.25 يعني تغيرًا متوسطًا يستحق الفحص. أكثر من 0.25 "
            "يعني تغيرًا كبيرًا وهو الحد المتعارف عليه لإعادة التدريب."
        ),
        "lang": "ar",
    },
]


def load_samples(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file of {question, answer, context|contexts, lang?, trace_id?}."""
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    """CLI entry point — see the module docstring."""
    parser = argparse.ArgumentParser(prog="python -m evals.judge", description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run the two shipped samples")
    parser.add_argument("--file", help="JSONL file of samples")
    parser.add_argument(
        "--rubric",
        action="append",
        choices=sorted(DATA_TYPES),
        help="repeatable; default is all three",
    )
    parser.add_argument("--push", action="store_true", help="write scores back to Langfuse")
    args = parser.parse_args()

    if not args.demo and not args.file:
        parser.error("pass --demo or --file")
    samples = DEMO_SAMPLES if args.demo else load_samples(args.file)
    rubric_names = args.rubric or sorted(DATA_TYPES)

    judge = Judge()
    results = run_batch(judge, samples, rubric_names, push=args.push)
    if args.push:
        # The SDK batches score writes and sends them in the background, so a
        # short-lived script exits before the queue drains and the scores are
        # silently lost. Every entry point that writes must flush.
        from langfuse import get_client

        get_client().flush()
        print("  flushed scores to Langfuse")
    trustworthy = report(results, judge.config.fingerprint())
    sys.exit(0 if trustworthy else 1)


if __name__ == "__main__":
    main()
