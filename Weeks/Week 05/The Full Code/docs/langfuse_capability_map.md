# Langfuse capability map — what to project, and in what order

Not a tour script. Every capability this session teaches is already demonstrated
in code that runs, and a second file that demonstrates them again would be a
second file to keep correct. This is the index into the code that exists.

Two files carry all of it:

- **[`ollama_langfuse_rag.py`](../ollama_langfuse_rag.py)** — one traced RAG
  pipeline, hand-instrumented. Read this to learn *what an observation is*.
- **[`langfuse_workload.py`](../langfuse_workload.py)** — the same primitives
  driven in bulk. Read this to learn *what to run and how often*, which is the
  part a demo usually skips and a production system never gets to skip.

> **Why no `langfuse.openai` wrapper anywhere.** The drop-in wrapper is one line
> and traces everything automatically, which is exactly why it teaches nothing.
> Doing it by hand shows what a generation observation actually consists of —
> model, parameters, input, output, token usage, cost. Use the wrapper in real
> code, once this is clear. See `ollama_langfuse_rag.py:28`.

---

## The map

| Capability | Where | Lines |
|---|---|---|
| **Masking PII at the SDK boundary** | `ollama_langfuse_rag.py` | `105-135` |
| Client constructed with `mask=` | `ollama_langfuse_rag.py` | `135` |
| **Nested tracing** — retrieval as its own span | `ollama_langfuse_rag.py` | `212-258` |
| `@observe` with `as_type="retriever"` | `ollama_langfuse_rag.py` | `212` |
| **Token usage** captured from the model response | `ollama_langfuse_rag.py` | `260-276` |
| **Cost** — `usage_details` + `cost_details` | `ollama_langfuse_rag.py` | `277-286` |
| The instrumented generation itself | `ollama_langfuse_rag.py` | `287-370` |
| **Prompt management** — fetch, compile, link | `ollama_langfuse_rag.py` | `371-395` |
| **`propagate_attributes`** — trace-level metadata | `ollama_langfuse_rag.py` | `410`, `834` |
| Language tagging (`lang:ar` / `lang:en`) | `ollama_langfuse_rag.py` | `396-425` |
| `as_type="tool"` and `as_type="agent"` | `ollama_langfuse_rag.py` | `500-554` |
| **Time-to-first-token**, recorded separately | `ollama_langfuse_rag.py` | `555-600` |
| `as_type="evaluator"` — the judge | `ollama_langfuse_rag.py` | `635-664` |
| **Four score types** (NUMERIC/BOOLEAN/CATEGORICAL) | `ollama_langfuse_rag.py` | `665-700` |
| **Errors** captured as ERROR, not vanished | `ollama_langfuse_rag.py` | `706-730` |
| **Pricing a self-hosted model** | `langfuse_workload.py` | `70-100` |
| **Prompt versions** — two live, traffic split | `langfuse_workload.py` | `256-284` |
| `as_type="guardrail"` | `langfuse_workload.py` | `315-333` |
| **Human-feedback scores** (the third source) | `langfuse_workload.py` | `382-432` |
| **Sessions** — a conversation as one thread | `langfuse_workload.py` | `619-728` |
| **Datasets** | `langfuse_workload.py` | `729-770` |
| **Experiments** — one run per prompt version | `langfuse_workload.py` | `852-885` |
| Custom evaluators for an experiment | `langfuse_workload.py` | `771-851` |
| **Judge scores written back** (`judge_*`) | `evals/judge.py` | `push_score` |
| **RAGAS scores written back** (`ragas_*`) | `evals/run_ragas.py` | `push_scores` |
| **Cost report** — min/max week | `tools/cost_report.py` | whole file |

---

## Chapter 4, in projection order

Twenty minutes. Project the file on one half of the screen and the Langfuse UI
on the other; every demo prints the trace URL it just created.

1. **Run it first.** `python ollama_langfuse_rag.py --demo rag`. Open the trace.
   Let them see the waterfall before any code — retrieve, generate, judge, one
   trace, nested. Everything after this is "how".

2. **`ollama_langfuse_rag.py:212-258` — the retrieval span.** The key line is
   `as_type="retriever"`, and the reason is in the docstring: *most reported
   hallucinations are retrieval failures, and you can only see that if retrieval
   is its own span*. The `zero_hits` and `lang` metadata are what incident 09 is
   later diagnosed from.

3. **`287-370` — the generation.** Model, parameters, input, output, usage,
   cost. This is what the one-line wrapper would have hidden. Point at
   `usage_details` and `cost_details` explicitly.

4. **`277-286` and `langfuse_workload.py:70-100` — why cost is not zero.**
   Langfuse prices a generation by matching its model name against a price list,
   and there is no entry for `llama3.1:8b`. The dashboard reads `$0.00` and
   everyone concludes inference is free. The honest unit for self-hosted is
   GPU-hours × instance price ÷ tokens produced. Then run `make cost`.

5. **`665-700` — the four score types**, and *why* they are different: NUMERIC
   for charts and thresholds, BOOLEAN for gates, CATEGORICAL for filtering.

6. **`langfuse_workload.py:619-728` — one conversation, one session.** Show the
   Sessions tab. Then `256-284`: two prompt versions live at once, traffic split
   between them.

7. **Inject incident 07.** The production label moves to the degraded prompt. No
   deploy, no restart, so the deploy timeline is empty — which is why it takes an
   hour to find. Students filter traces by prompt version, see faithfulness split,
   and move the label back. Two-minute rollback, no deploy.

8. **`105-135` — masking**, last and briefly. One line at construction, and the
   only thing on this page that cannot be retrofitted: a value that reached
   ClickHouse unmasked has been written, backed up, and possibly indexed.

---

## What is deliberately *not* here

- **`langfuse.openai`** — see the note above.
- **A separate tour script** — the capabilities are demonstrated by code that
  does real work. A demo that exists only to be demoed rots quietly, because
  nothing fails when it stops being true.
- **Cloud Langfuse** — the stack is self-hosted so the whole session runs
  offline after one pull, and so nobody's course data leaves the room.
