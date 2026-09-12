"""LLM observability with Langfuse, against a local Ollama model.

Unlike the other scripts in this session, this one RUNS. End to end:

    cp .env.example .env
    docker compose --profile langfuse up -d      # Langfuse on :3001
    ollama serve                                 # Ollama on :11434
    ollama pull llama3.1:8b
    pip install -e ".[llm]"
    python ollama_langfuse_rag.py                # → traces at http://localhost:3001

The domain is deliberately this session's own material: a small RAG assistant
that answers questions about drift detection using the notes in KNOWLEDGE_BASE.
So you can read an answer and judge whether it's actually grounded.

Four demos, each isolating one thing Langfuse gives you that a print statement
does not:

    rag     nested spans — retrieve → generate → judge, one trace. The
            LLM-as-judge and the NUMERIC/BOOLEAN/CATEGORICAL scores it writes
            run as part of this demo, not as a separate one.
    tools   an agent loop, every tool call its own observation
    stream  time-to-first-token recorded separately from total latency
    error   a failed call captured as ERROR rather than vanishing

Run one with `--demo rag`, or all four with no arguments.

Why hand-instrumented instead of `from langfuse.openai import OpenAI` against
Ollama's OpenAI-compatible endpoint: that wrapper is one line and traces
everything automatically, which is exactly why it teaches you nothing. Doing it
by hand shows what a "generation" observation actually consists of — model,
parameters, input, output, token usage, cost. Once that's clear, use the
auto-instrumentation in real code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import ollama
from langfuse import Langfuse, observe, propagate_attributes

# ══════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════
#
# The langfuse SDK reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY /
# LANGFUSE_BASE_URL from the environment — get_client() takes no arguments.
# .env.example ships the auto-provisioned keys the compose stack creates on
# first boot, so there is no UI setup step.
#
# Loading .env here is a convenience so `python ollama_langfuse_rag.py` works
# straight after `cp .env.example .env`, with no `export` step. It must happen
# BEFORE get_client() below, which snapshots the environment at construction.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # optional — export the variables yourself instead
    pass

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Tool calling needs a model trained for it — llama3.1 is; qwen2.5:0.5b is not.
CHAT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", CHAT_MODEL)

# Sent to Langfuse as `model_parameters` so a bad answer can be traced back to
# the settings that produced it. temperature=0 keeps the demo reproducible.
MODEL_PARAMS: dict[str, Any] = {
    "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0")),
    "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "400")),
    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096")),
}

# ── Cost ──────────────────────────────────────────────────────────────
# A local model has no per-token price, so USD cost is genuinely 0 — but the
# Langfuse cost UI is one of the main reasons to use it, and you want the wiring
# in place before you swap in a hosted model. Fill in real numbers here and the
# dashboards light up without touching the call sites.
#
# For self-hosted models the honest unit isn't $/token at all, it's
# GPU-hours × instance price ÷ tokens produced. Set OLLAMA_USD_PER_1K_OUTPUT to
# your amortised figure if you want that reflected.
USD_PER_1K_INPUT = float(os.getenv("OLLAMA_USD_PER_1K_INPUT", "0"))
USD_PER_1K_OUTPUT = float(os.getenv("OLLAMA_USD_PER_1K_OUTPUT", "0"))

# ── Masking: redact PII BEFORE it is sent, not after ──────────────────
# Applied at the SDK boundary, so a matched value never leaves the process.
# Redacting after ingestion is not redaction — it is deletion, and by then the
# value has been written to ClickHouse, backed up, and possibly indexed.
#
# The patterns are deliberately conservative. An over-eager mask that eats the
# answer text makes traces useless and gets switched off within a week, which
# leaves you with no masking at all.
_PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<EMAIL>"),
    (re.compile(r"(?<!\d)(?:\+20|0)1[0-2,5]\d{8}(?!\d)"), "<PHONE>"),  # Egyptian mobile
    (re.compile(r"(?<!\d)[23]\d{13}(?!\d)"), "<NATIONAL_ID>"),  # Egyptian national id
    (re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"), "<CARD>"),
]


def mask_pii(data: Any) -> Any:
    """Redact emails, phone numbers and national ids anywhere in a payload.

    Passed as `Langfuse(mask=...)` at construction, so it runs on every input,
    output and metadata value the SDK is about to send — including the ones you
    forgot were user-controlled. Recursive, because trace payloads are nested.
    """
    if isinstance(data, str):
        for pattern, replacement in _PII_PATTERNS:
            data = pattern.sub(replacement, data)
        return data
    if isinstance(data, dict):
        return {k: mask_pii(v) for k, v in data.items()}
    if isinstance(data, list):
        return [mask_pii(v) for v in data]
    return data


client = ollama.Client(host=OLLAMA_HOST)
# Constructed rather than get_client(): the mask can only be installed at
# construction, and it is the single highest-leverage line in this file for
# anyone who will run it against real user traffic.
langfuse = Langfuse(mask=mask_pii)


# ══════════════════════════════════════════════════════════════════════
#  A tiny knowledge base + a dependency-free retriever
# ══════════════════════════════════════════════════════════════════════
#
# Real RAG embeds documents and does vector search. This uses TF-IDF-weighted
# lexical overlap instead — not because it's better (it isn't), but because it
# keeps this file runnable with zero extra services and zero extra models.
#
# To make it a real vector store, replace score_document() with a cosine
# similarity over `ollama.embed(model="nomic-embed-text", input=...)`. Nothing
# else in this file changes — the retrieval span is already there to receive it.
# (Note: `ollama serve` must be started with embeddings enabled for that.)

# The corpus itself lives in evals/knowledge_base.py — 44 notes, each written
# in English and Arabic. It moved out of this file when session 4 added the
# judge-calibration chapter, which needs Arabic traffic to exist before it can
# measure how much worse a small judge is at Arabic than at English.
from evals.knowledge_base import (  # noqa: E402
    NOTES,
    detect_language,
    docs_for,
    index_is_asymmetric,
    index_tokens,
    query_tokens,
)

#: The corpus in each language. A query is answered from the corpus matching its
#: own language — mixing them would let an Arabic question retrieve English
#: notes, which hides exactly the failure incident 09 is about.
CORPORA: dict[str, list[dict[str, str]]] = {lang: docs_for(lang) for lang in ("en", "ar")}

#: Kept under its old name: langfuse_workload.py and this module's docstring
#: both refer to KNOWLEDGE_BASE, and English is still the default corpus.
KNOWLEDGE_BASE: list[dict[str, str]] = CORPORA["en"]
_N_DOCS = len(NOTES)


def _build_index(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Tokenize a corpus once and precompute document frequency."""
    tokens = {d["id"]: index_tokens(f"{d['title']} {d['text']}") for d in docs}
    return {
        "tokens": tokens,
        "freq": Counter(tok for toks in tokens.values() for tok in set(toks)),
    }


#: Below this, a document is not considered a hit at all. See retrieve().
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "1.2"))

#: Built once at import, per language.
_INDEX: dict[str, dict[str, Any]] = {lang: _build_index(docs) for lang, docs in CORPORA.items()}


def _tokenize(text: str) -> list[str]:
    """Kept for backwards compatibility — query-side tokenization."""
    return query_tokens(text)


def score_document(q_tokens: list[str], doc_id: str, lang: str = "en") -> float:
    """IDF-weighted overlap, length-normalised. Swap for cosine similarity."""
    index = _INDEX[lang]
    doc_counts = Counter(index["tokens"][doc_id])
    n_docs = len(index["tokens"])
    score = 0.0
    for tok in set(q_tokens):
        if tok not in doc_counts:
            continue
        # Rare words in the corpus discriminate; words in every doc do not.
        idf = math.log((n_docs + 1) / (index["freq"][tok] + 1)) + 1.0
        score += idf * (1 + math.log(doc_counts[tok]))
    return score / math.sqrt(len(index["tokens"][doc_id]) or 1)


@observe(name="retrieve", as_type="retriever")
def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Fetch the k most relevant documents, from the query's own language.

    `as_type="retriever"` is not cosmetic — Langfuse renders retrieval
    observations differently from generations, and being able to filter traces
    by retrieval quality is how you find out that a bad answer was actually a
    bad *retrieval*. Most "the LLM hallucinated" bugs are this.
    """
    lang = detect_language(query)
    q_tokens = query_tokens(query)
    # A retriever with no minimum score ALWAYS returns k documents, so it can
    # never report that it failed — "no relevant documents" and "here are the
    # three least irrelevant documents" look identical downstream, and the model
    # answers confidently from whatever came back. A floor is what turns a bad
    # retrieval into an observable event.
    scored = sorted(
        ((score_document(q_tokens, d["id"], lang), d) for d in CORPORA[lang]),
        key=lambda pair: pair[0],
        reverse=True,
    )
    hits = [
        {"id": d["id"], "title": d["title"], "text": d["text"], "score": round(s, 4)}
        for s, d in scored[:k]
        if s >= MIN_RETRIEVAL_SCORE
    ]

    # @observe already captured the args and the return value. This adds the
    # numbers you would actually alert on: zero hits, or a top score near zero,
    # means the retriever found nothing and the model is about to answer from
    # memory. Both are recorded PER LANGUAGE, because a global retrieval metric
    # cannot show one language collapsing while the other stays perfect.
    langfuse.update_current_span(
        metadata={
            "top_score": hits[0]["score"] if hits else 0.0,
            "retrieved_ids": [h["id"] for h in hits],
            "n_hits": len(hits),
            "zero_hits": not hits,
            "lang": lang,
            "corpus_size": len(CORPORA[lang]),
            "retriever": "tfidf-lexical",
            "index_asymmetric": index_is_asymmetric(),
        }
    )
    return hits


# ══════════════════════════════════════════════════════════════════════
#  The instrumented Ollama call — the core of this file
# ══════════════════════════════════════════════════════════════════════


def _usage_from_ollama(response: Any) -> dict[str, int]:
    """Map Ollama's counters onto Langfuse's usage schema.

    Ollama returns `prompt_eval_count` (tokens read) and `eval_count` (tokens
    generated). Langfuse expects the keys `input` / `output` / `total`; get
    these names wrong and the token columns silently stay empty, which is the
    single most common mistake when hand-instrumenting a provider.
    """
    prompt_tokens = int(getattr(response, "prompt_eval_count", 0) or 0)
    completion_tokens = int(getattr(response, "eval_count", 0) or 0)
    return {
        "input": prompt_tokens,
        "output": completion_tokens,
        "total": prompt_tokens + completion_tokens,
    }


def _cost_from_usage(usage: dict[str, int]) -> dict[str, float]:
    """Same shape as usage, in dollars. Zero for a local model — see the top."""
    return {
        "input": usage["input"] / 1000 * USD_PER_1K_INPUT,
        "output": usage["output"] / 1000 * USD_PER_1K_OUTPUT,
        "total": (
            usage["input"] / 1000 * USD_PER_1K_INPUT + usage["output"] / 1000 * USD_PER_1K_OUTPUT
        ),
    }


def chat(
    messages: list[dict[str, Any]],
    *,
    name: str = "ollama-chat",
    model: str = CHAT_MODEL,
    tools: list[dict[str, Any]] | None = None,
    prompt: Any = None,
    fmt: str | None = None,
) -> Any:
    """One traced call to Ollama.

    Everything Langfuse knows about an LLM call is set here, and it's worth
    seeing the whole list in one place:

      model / model_parameters  what produced the output — the first thing you
                                need when a regression appears after a config change
      input                     the full message list, system prompt included
      output                    the completion
      usage_details             token counts, which drive the cost and volume charts
      cost_details              dollars, if you priced the model
      prompt                    the managed prompt version used, so you can
                                compare quality across prompt versions

    `start_as_current_observation` is a context manager: the observation is
    opened on entry, timed, and closed on exit — including on an exception,
    which is how the `error` demo lands in the UI as a failed span rather than
    as nothing at all.
    """
    with langfuse.start_as_current_observation(
        name=name,
        as_type="generation",
        model=model,
        model_parameters=MODEL_PARAMS,
        input=messages,
        prompt=prompt,
    ) as generation:
        options = dict(MODEL_PARAMS)
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "options": options}
        if tools:
            kwargs["tools"] = tools
        if fmt:
            kwargs["format"] = fmt

        response = client.chat(**kwargs)

        usage = _usage_from_ollama(response)
        generation.update(
            output=response.message.content or response.message.model_dump().get("tool_calls"),
            usage_details=usage,
            cost_details=_cost_from_usage(usage),
            metadata={
                # Ollama reports nanoseconds. load_duration is model load time —
                # it dominates the first call after a cold start and will
                # otherwise look like the model got mysteriously slower.
                "load_ms": round((response.load_duration or 0) / 1e6, 1),
                "prompt_eval_ms": round((response.prompt_eval_duration or 0) / 1e6, 1),
                "eval_ms": round((response.eval_duration or 0) / 1e6, 1),
                "done_reason": response.done_reason,
            },
        )
        return response


# ══════════════════════════════════════════════════════════════════════
#  Prompt management
# ══════════════════════════════════════════════════════════════════════
#
# The point of storing the prompt in Langfuse rather than in this file: you can
# edit it in the UI, label a version `production`, and see quality scores broken
# down by prompt version — without a redeploy. The generation above receives
# `prompt=` so every trace records which version answered.

PROMPT_NAME = "session4-drift-assistant"
PROMPT_TEXT = """You are a monitoring assistant for an ML platform team.
Answer the question using ONLY the context below. If the context does not
contain the answer, say "I don't know from the provided context." — do not
draw on outside knowledge. Be concise: three sentences at most.

Context:
{{context}}

Question: {{question}}"""

#: Incident 07. The same prompt with the grounding sentence removed and a
#: friendlier tone — the kind of edit that gets made because the answers "read
#: better", by someone who has never seen a faithfulness chart. Nothing else
#: changes: no code, no deploy, no restart, so the deploy timeline stays empty
#: and the only trace of the change is the prompt version on each generation.
DEGRADED_PROMPT_TEXT = """You are a friendly monitoring assistant for an ML
platform team. Answer the question below, keeping it warm and helpful.

Context:
{{context}}

Question: {{question}}"""


def ensure_prompt() -> Any:
    """Fetch the prompt carrying the production label, creating it on first run.

    `label="production"` is the indirection that matters: this code never names
    a version number, so promoting a new prompt is a label move in the UI, not a
    code change. That is also precisely why incident 07 is hard to find — the
    label below is the ONLY thing that changes, and it leaves no deploy row.
    """
    from incidents import state

    label = state.flag("PROMPT_PRODUCTION_LABEL") or "production"
    try:
        return langfuse.get_prompt(PROMPT_NAME, label=label, cache_ttl_seconds=60)
    except Exception:
        # First run for this label. "degraded" is created on demand so incident
        # 07 works on a fresh instance without a separate seeding step.
        is_degraded = label != "production"
        return langfuse.create_prompt(
            name=PROMPT_NAME,
            prompt=DEGRADED_PROMPT_TEXT if is_degraded else PROMPT_TEXT,
            type="text",
            labels=[label],
            commit_message=(
                "Softer tone, grounding instruction dropped"
                if is_degraded
                else "Initial version from ollama_langfuse_rag.py"
            ),
        )


# ══════════════════════════════════════════════════════════════════════
#  Demo 1 — RAG: one trace, three nested observations
# ══════════════════════════════════════════════════════════════════════


@observe(name="rag-pipeline")
def rag_pipeline(question: str) -> dict[str, Any]:
    """retrieve → build prompt → generate, as a single trace.

    Because @observe nests automatically via OpenTelemetry context, the spans
    below become children of this one with no plumbing. That nesting is the
    whole value: when an answer is wrong you can see whether retrieval missed,
    the prompt was malformed, or the model ignored good context.
    """
    # Tag the whole trace with the question's language. Every quality number in
    # this repo is reported per language, and that is only possible if the tag
    # is on the trace at write time — you cannot retrofit a dimension onto
    # traces you have already collected. propagate_attributes WRAPS the work
    # because it applies to spans created inside the block.
    lang = detect_language(question)
    with propagate_attributes(tags=[f"lang:{lang}"], metadata={"lang": lang}):
        docs = retrieve(question, k=3)

        prompt_client = ensure_prompt()
        context = "\n\n".join(f"[{d['id']}] {d['title']}: {d['text']}" for d in docs)
        compiled = prompt_client.compile(context=context, question=question)

        response = chat(
            [{"role": "user", "content": compiled}],
            name="generate-answer",
            prompt=prompt_client,  # links trace → prompt version
        )
        answer = response.message.content.strip()

    return {"question": question, "answer": answer, "docs": docs, "lang": lang}


# ══════════════════════════════════════════════════════════════════════
#  Demo 2 — tool calling: an agent loop where every step is observable
# ══════════════════════════════════════════════════════════════════════
#
# Stand-ins for real monitoring queries. In production these would hit the
# Prometheus HTTP API — `feature_psi_score{feature="distance_km"}` and
# `histogram_quantile(0.95, ...)` are exactly the series from section 5 of the
# README.

FAKE_PSI = {"distance_km": 0.31, "passengers": 0.04, "hour_of_day": 0.12}
FAKE_LATENCY_P95_MS = {"/predict": 412.0, "/batch_predict": 1830.0}


@observe(name="tool:get_psi", as_type="tool")
def get_psi(feature: str) -> dict[str, Any]:
    score = FAKE_PSI.get(feature)
    if score is None:
        return {"error": f"unknown feature {feature!r}", "known": sorted(FAKE_PSI)}
    verdict = "retrain" if score > 0.25 else "investigate" if score > 0.10 else "stable"
    return {"feature": feature, "psi": score, "verdict": verdict}


@observe(name="tool:get_latency_p95", as_type="tool")
def get_latency_p95(endpoint: str) -> dict[str, Any]:
    ms = FAKE_LATENCY_P95_MS.get(endpoint)
    if ms is None:
        return {"error": f"unknown endpoint {endpoint!r}", "known": sorted(FAKE_LATENCY_P95_MS)}
    return {"endpoint": endpoint, "p95_ms": ms, "breaches_slo": ms > 500}


TOOL_IMPLS = {"get_psi": get_psi, "get_latency_p95": get_latency_p95}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_psi",
            "description": "Current Population Stability Index drift score for one feature.",
            "parameters": {
                "type": "object",
                "properties": {
                    "feature": {
                        "type": "string",
                        "description": "Feature name, e.g. distance_km",
                        "enum": sorted(FAKE_PSI),
                    }
                },
                "required": ["feature"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_latency_p95",
            "description": "Current p95 latency in milliseconds for one API endpoint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "API path, e.g. /predict",
                        "enum": sorted(FAKE_LATENCY_P95_MS),
                    }
                },
                "required": ["endpoint"],
            },
        },
    },
]


@observe(name="monitoring-agent", as_type="agent")
def monitoring_agent(question: str, max_turns: int = 4) -> dict[str, Any]:
    """A model that can query monitoring state, then answer.

    Each iteration is a generation; each tool the model picks is its own
    observation, nested under this agent. That structure answers the two
    questions you always have about a misbehaving agent: which tool did it call
    with what arguments, and how many turns did it burn getting there.
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are an ML monitoring assistant. Use the provided tools to "
                "look up live drift and latency numbers before answering. "
                "PSI above 0.25 means retrain. Answer in two sentences."
            ),
        },
        {"role": "user", "content": question},
    ]

    tool_calls_made: list[dict[str, Any]] = []
    for turn in range(max_turns):
        response = chat(messages, name=f"agent-turn-{turn + 1}", tools=TOOL_SCHEMAS)
        message = response.message
        messages.append(message.model_dump())

        if not message.tool_calls:
            langfuse.update_current_span(
                metadata={"turns_used": turn + 1, "tool_calls": tool_calls_made}
            )
            return {"answer": (message.content or "").strip(), "tool_calls": tool_calls_made}

        for call in message.tool_calls:
            fn_name = call.function.name
            args = dict(call.function.arguments or {})
            impl = TOOL_IMPLS.get(fn_name)
            result = impl(**args) if impl else {"error": f"no such tool {fn_name!r}"}
            tool_calls_made.append({"tool": fn_name, "args": args, "result": result})
            # Feeding the result back as a `tool` message is what closes the loop.
            messages.append({"role": "tool", "name": fn_name, "content": json.dumps(result)})

    langfuse.update_current_span(
        level="WARNING",
        status_message=f"hit max_turns={max_turns} without a final answer",
        metadata={"tool_calls": tool_calls_made},
    )
    return {"answer": "", "tool_calls": tool_calls_made, "exhausted": True}


# ══════════════════════════════════════════════════════════════════════
#  Demo 3 — streaming, and why time-to-first-token is its own metric
# ══════════════════════════════════════════════════════════════════════


@observe(name="streaming-answer")
def streaming_answer(question: str) -> str:
    """Stream tokens, recording TTFT separately from total duration.

    `completion_start_time` is the field that makes this worth doing. Total
    latency is what your infrastructure pays; time-to-first-token is what the
    user actually perceives. A model that streams the first token in 200ms and
    finishes in 8s feels fast; one that thinks for 6s then dumps everything in
    2s feels broken. Only one of those is visible in an averaged latency chart.
    """
    messages = [{"role": "user", "content": question}]

    with langfuse.start_as_current_observation(
        name="ollama-stream",
        as_type="generation",
        model=CHAT_MODEL,
        model_parameters=MODEL_PARAMS,
        input=messages,
    ) as generation:
        chunks: list[str] = []
        first_token_at: datetime | None = None
        started = time.perf_counter()
        final: Any = None

        for chunk in client.chat(
            model=CHAT_MODEL, messages=messages, options=dict(MODEL_PARAMS), stream=True
        ):
            piece = chunk.message.content or ""
            if piece and first_token_at is None:
                first_token_at = datetime.now(timezone.utc)
                ttft_ms = (time.perf_counter() - started) * 1000
            if piece:
                chunks.append(piece)
                print(piece, end="", flush=True)
            if chunk.done:
                # Usage counters only arrive on the final chunk.
                final = chunk
        print()

        text = "".join(chunks)
        usage = _usage_from_ollama(final) if final is not None else {}
        generation.update(
            output=text,
            completion_start_time=first_token_at,
            usage_details=usage or None,
            cost_details=_cost_from_usage(usage) if usage else None,
            metadata={
                "ttft_ms": round(ttft_ms, 1) if first_token_at else None,
                "total_ms": round((time.perf_counter() - started) * 1000, 1),
                "chunks": len(chunks),
            },
        )
        return text


# ══════════════════════════════════════════════════════════════════════
#  Demo 4 — evaluation: scores are what turn traces into a quality signal
# ══════════════════════════════════════════════════════════════════════
#
# Traces alone tell you what happened. Scores tell you whether it was any good,
# and they're the only thing you can chart, alert on, or compare across prompt
# versions. This is the direct analogue of the rest of session 4: for a
# regression model you track PSI over time; for an LLM you track scores over
# time. Same shape of problem, different data type.

JUDGE_TEMPLATE = """You are a strict evaluator. Score the ANSWER against the CONTEXT.

Return ONLY a JSON object with these keys:
  "faithfulness": 0.0-1.0, is every claim in the answer supported by the context?
  "relevance":    0.0-1.0, does the answer address the question?
  "reason":       one short sentence.

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}"""


@observe(name="llm-judge", as_type="evaluator")
def judge_answer(question: str, answer: str, docs: list[dict[str, Any]]) -> dict[str, Any]:
    """LLM-as-judge over one answer.

    `format="json"` constrains Ollama's decoding to valid JSON, which removes
    most of the parsing pain. The bare `except` below is still deliberate: a
    judge that fails must never take down the pipeline it is grading, so a
    failed evaluation degrades to a null score rather than an exception.
    """
    context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in docs)
    response = chat(
        [
            {
                "role": "user",
                "content": JUDGE_TEMPLATE.format(context=context, question=question, answer=answer),
            }
        ],
        name="judge-generation",
        model=JUDGE_MODEL,
        fmt="json",
    )
    try:
        verdict = json.loads(response.message.content)
        faithfulness = float(verdict.get("faithfulness", 0.0))
        relevance = float(verdict.get("relevance", 0.0))
        reason = str(verdict.get("reason", ""))[:500]
    except Exception as exc:  # noqa: BLE001 — see docstring
        langfuse.update_current_span(
            level="WARNING", status_message=f"unparseable judge output: {exc}"
        )
        return {"faithfulness": None, "relevance": None, "reason": "judge failed"}

    return {"faithfulness": faithfulness, "relevance": relevance, "reason": reason}


def attach_scores(result: dict[str, Any], verdict: dict[str, Any]) -> None:
    """Write scores onto the CURRENT trace.

    Three kinds, because Langfuse treats them differently:
      NUMERIC     — charts and thresholds (this is what you alert on)
      BOOLEAN     — pass/fail gates
      CATEGORICAL — slice-and-filter dimensions
    """
    if verdict["faithfulness"] is not None:
        langfuse.score_current_trace(
            name="faithfulness",
            value=verdict["faithfulness"],
            data_type="NUMERIC",
            comment=verdict["reason"],
        )
        langfuse.score_current_trace(
            name="relevance", value=verdict["relevance"], data_type="NUMERIC"
        )
        langfuse.score_current_trace(
            name="grounded",
            value=verdict["faithfulness"] >= 0.7,
            data_type="BOOLEAN",
        )

    # A cheap deterministic check alongside the LLM judge. Heuristics like this
    # are underrated: they cost nothing, never flake, and catch the failure mode
    # that matters most in RAG — the model answering with nothing retrieved.
    top_score = result["docs"][0]["score"] if result["docs"] else 0.0
    langfuse.score_current_trace(
        name="retrieval_hit",
        value="hit" if top_score > 0.5 else "weak" if top_score > 0.1 else "miss",
        data_type="CATEGORICAL",
    )


# ══════════════════════════════════════════════════════════════════════
#  Demo 5 — failures are observations too
# ══════════════════════════════════════════════════════════════════════


@observe(name="failing-call")
def failing_call() -> str:
    """Deliberately call a model that isn't installed.

    An LLM app fails in ways a traceback doesn't capture: a rate limit, a
    content filter, a context-length overflow, a model that was deprecated
    yesterday. Marking the observation ERROR with the provider's own message
    means the failure shows up next to the successful traces, filterable, with
    the input that caused it — instead of only in a log file nobody greps.
    """
    bogus = "this-model-does-not-exist:latest"
    messages = [{"role": "user", "content": "Will this work?"}]
    with langfuse.start_as_current_observation(
        name="doomed-generation", as_type="generation", model=bogus, input=messages
    ) as generation:
        try:
            client.chat(model=bogus, messages=messages)
        except Exception as exc:  # noqa: BLE001 — the point of the demo
            generation.update(
                level="ERROR",
                status_message=str(exc)[:500],
                output={"error": type(exc).__name__},
            )
            return f"failed as expected: {type(exc).__name__}"
    return "unexpectedly succeeded"


# ══════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════


def preflight() -> None:
    """Fail loudly and specifically, before burning a minute on model load."""
    try:
        installed = {m.model for m in client.list().models}
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"✗ Ollama unreachable at {OLLAMA_HOST}: {exc}\n  Start it with: ollama serve")

    for model in {CHAT_MODEL, JUDGE_MODEL}:
        if model not in installed:
            sys.exit(f"✗ Model {model!r} not installed.\n  Pull it with: ollama pull {model}")

    if not langfuse.auth_check():
        sys.exit(
            "✗ Langfuse auth failed.\n"
            "  Check LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL,\n"
            "  and that the stack is up: docker compose --profile langfuse up -d"
        )
    print(f"✓ Ollama {OLLAMA_HOST} · model {CHAT_MODEL}")
    print(f"✓ Langfuse {os.getenv('LANGFUSE_BASE_URL', 'http://localhost:3001')}\n")


def _trace_link(label: str) -> None:
    """Print a clickable link to the trace just produced."""
    trace_id = langfuse.get_current_trace_id()
    url = langfuse.get_trace_url(trace_id=trace_id) if trace_id else None
    print(f"  ↳ {label} trace: {url or trace_id}")


@observe(name="demo-rag")
def demo_rag(question: str) -> None:
    result = rag_pipeline(question)
    print(f"Q: {question}\nA: {result['answer']}")
    print(f"   retrieved: {[d['id'] for d in result['docs']]}")

    verdict = judge_answer(question, result["answer"], result["docs"])
    attach_scores(result, verdict)
    print(
        f"   judge: faithfulness={verdict['faithfulness']} "
        f"relevance={verdict['relevance']} — {verdict['reason']}"
    )

    # In a real app this is a thumbs-up/down from the UI, arriving seconds or
    # hours later. It's the score that matters most, because it's the only one
    # that isn't a model grading a model.
    langfuse.score_current_trace(
        name="user_feedback", value=1, data_type="NUMERIC", comment="simulated thumbs-up"
    )
    _trace_link("rag")


@observe(name="demo-tools")
def demo_tools(question: str) -> None:
    result = monitoring_agent(question)
    print(f"Q: {question}\nA: {result['answer']}")
    for call in result["tool_calls"]:
        print(f"   tool {call['tool']}({call['args']}) → {call['result']}")
    _trace_link("tools")


@observe(name="demo-stream")
def demo_stream(question: str) -> None:
    print(f"Q: {question}\nA: ", end="", flush=True)
    streaming_answer(question)
    _trace_link("stream")


@observe(name="demo-error")
def demo_error() -> None:
    print(failing_call())
    _trace_link("error")


DEMOS = ("rag", "tools", "stream", "error")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo", choices=("all", *DEMOS), default="all")
    parser.add_argument(
        "--question", default="When should I retrain based on PSI, and why not use KS?"
    )
    parser.add_argument("--user-id", default="student-01")
    parser.add_argument("--session-id", default="session-4-demo")
    args = parser.parse_args()

    preflight()

    selected = DEMOS if args.demo == "all" else (args.demo,)

    # ── propagate_attributes: trace-level metadata for everything inside ──
    #
    # This is the v4 replacement for v2's langfuse_context.update_current_trace().
    # It must WRAP the calls, not sit inside them, because it applies to spans
    # created within the block.
    #
    # session_id is what groups a multi-turn conversation into one thread in the
    # UI; user_id is what lets you answer "what is this specific user seeing?"
    # when they complain. Both are worth wiring on day one — retrofitting them
    # means the traces you already have stay unattributed forever.
    with propagate_attributes(
        user_id=args.user_id,
        session_id=args.session_id,
        tags=["session-4", "ollama", "demo"],
        metadata={"chat_model": CHAT_MODEL, "judge_model": JUDGE_MODEL},
        version="1.0.0",
    ):
        for demo in selected:
            print(f"\n{'═' * 70}\n  {demo}\n{'═' * 70}")
            if demo == "rag":
                demo_rag(args.question)
            elif demo == "tools":
                demo_tools("Is distance_km drifting, and is /predict breaching its latency SLO?")
            elif demo == "stream":
                demo_stream("In two sentences, why does concept drift need ground truth?")
            elif demo == "error":
                demo_error()

    # Short-lived scripts MUST flush. The SDK batches spans on a background
    # thread; without this the process exits and the last traces are lost — the
    # classic "my code ran but Langfuse is empty" bug.
    langfuse.flush()
    print(f"\n✓ Flushed. Open {os.getenv('LANGFUSE_BASE_URL', 'http://localhost:3001')}")


if __name__ == "__main__":
    main()
