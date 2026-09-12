"""Build the frozen evaluation set, and refuse to run if it has changed.

    python -m evals.build_testset            # build testset_v1.jsonl
    python -m evals.build_testset --stats     # describe the frozen file

Sixty items, balanced across English and Arabic, from three sources:

  synthetic     generated from the knowledge base — broad coverage, easy items
  mined         the lowest-scoring questions from seeded production traces —
                the ones the system actually struggles with
  adversarial   hand-written: questions with no answer in the corpus, questions
                that need two notes, and ambiguous questions

The adversarial block is the part that matters. A testset built only from things
the system already does well measures nothing, and the no-answer items are the
only way to find out whether the model refuses or confabulates — which is the
single most important thing to know about a RAG system.

**Why it is frozen.** A quality gate compares runs over time. If the testset
changes between runs, a score movement could be the model or could be the test,
and you cannot tell which. The loader checks a content hash and refuses to run
against a modified file rather than silently producing an incomparable number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from evals.knowledge_base import NOTES, detect_language

TESTSET = Path(__file__).resolve().parent / "testset_v1.jsonl"
HASH_FILE = Path(__file__).resolve().parent / "testset_v1.sha256"

#: Questions whose answer is deliberately NOT in the corpus. The right response
#: is a refusal; anything else is a confabulation, and no other item type in the
#: set can tell those two apart.
NO_ANSWER = [
    ("en", "What is the current p99 latency of the production ride-duration API?"),
    ("en", "How many rides did the Alexandria launch serve in its first week?"),
    ("en", "Which cloud provider hosts this course's MLflow registry?"),
    ("en", "What is the salary band for an MLOps engineer in Cairo?"),
    ("ar", "ما هو زمن الاستجابة عند المئين 99 لواجهة التنبؤ في الإنتاج حاليًا؟"),
    ("ar", "كم عدد الرحلات التي خدمتها الإسكندرية في أسبوعها الأول؟"),
    ("ar", "ما هي ميزانية الفريق لهذا الربع؟"),
    ("ar", "من هو المهندس المسؤول عن نوبة المناداة هذا الأسبوع؟"),
]

#: Each needs TWO notes to answer correctly. A retriever with k=1, or one that
#: returns three copies of the same idea, fails these and passes everything else.
TWO_NOTE = [
    ("en", "Why is PSI preferred to the KS test for setting an alert threshold?"),
    ("en", "How do late labels affect when the retrain gate can fire?"),
    ("en", "Why does a cardinality explosion make dashboards blank rather than noisy?"),
    ("en", "How are faithfulness and judge calibration related?"),
    ("ar", "لماذا يُفضَّل مؤشر PSI على اختبار KS عند تحديد حد التنبيه؟"),
    ("ar", "كيف تؤثر التسميات المتأخرة على توقيت عمل بوابة إعادة التدريب؟"),
    ("ar", "ما العلاقة بين الأمانة للسياق ومعايرة الحَكَم؟"),
    ("ar", "لماذا يجب فصل مقاييس الجودة حسب اللغة والمدينة معًا؟"),
]

#: Under-specified on purpose. A good answer asks which one, or answers both and
#: says so; a bad one silently picks whichever note retrieval happened to rank
#: first, which is indistinguishable from a correct answer unless you look.
AMBIGUOUS = [
    ("en", "What is the threshold?"),
    ("en", "Should I retrain?"),
    ("en", "Is 0.25 good or bad?"),
    ("ar", "ما هو الحد المسموح؟"),
    ("ar", "هل يجب أن أعيد التدريب؟"),
    ("ar", "هل هذا الرقم جيد؟"),
]


def synthetic(n: int, seed: int = 11) -> list[dict[str, Any]]:
    """Straightforward questions generated from the notes, half in each language."""
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    for lang in ("en", "ar"):
        notes = list(NOTES)
        rng.shuffle(notes)
        for note in notes[: n // 2]:
            title, text = note[lang]
            question = (
                f"What does {title} mean and when does it matter?"
                if lang == "en"
                else f"ما معنى {title} ومتى يكون مهمًا؟"
            )
            items.append(
                {
                    "question": question,
                    "reference": text,
                    "note_id": note["id"],
                    "lang": lang,
                    "source": "synthetic",
                    "expects_refusal": False,
                }
            )
    return items


def mined(path: Path, limit: int = 10) -> list[dict[str, Any]]:
    """Lowest-scoring questions from seeded traces, if a mined file exists.

    Kept optional so the testset can be rebuilt with no Langfuse running. In a
    real setup this is the most valuable third of the set: it is the only source
    that reflects what users actually ask rather than what you imagined.
    """
    if not path.exists():
        return []
    with open(path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rows.sort(key=lambda r: r.get("score", 1.0))
    return [
        {
            "question": r["question"],
            "reference": r.get("reference", ""),
            "note_id": r.get("note_id", ""),
            "lang": r.get("lang") or detect_language(r["question"]),
            "source": "mined",
            "expects_refusal": False,
        }
        for r in rows[:limit]
    ]


def adversarial() -> list[dict[str, Any]]:
    """The hand-written block: no-answer, two-note and ambiguous questions."""
    items = []
    for kind, rows, refusal in (
        ("no_answer", NO_ANSWER, True),
        ("two_note", TWO_NOTE, False),
        ("ambiguous", AMBIGUOUS, False),
    ):
        for lang, question in rows:
            items.append(
                {
                    "question": question,
                    "reference": "",
                    "note_id": "",
                    "lang": lang,
                    "source": f"adversarial:{kind}",
                    "expects_refusal": refusal,
                }
            )
    return items


def content_hash(items: list[dict[str, Any]]) -> str:
    """Stable hash over the questions — the identity of the measurement."""
    payload = "\n".join(sorted(i["question"] for i in items))
    return hashlib.sha256(payload.encode()).hexdigest()


def build(target: int = 60) -> list[dict[str, Any]]:
    """Assemble the set: adversarial first, then synthetic up to `target`."""
    items = adversarial()
    items += mined(TESTSET.with_name("mined_questions.jsonl"))
    items += synthetic(max(target - len(items), 0))
    for i, item in enumerate(items):
        item["id"] = f"{item['lang']}-{i:03d}"
    return items[:target]


def write(items: list[dict[str, Any]]) -> None:
    """Freeze the set and its hash."""
    with open(TESTSET, "w") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    digest = content_hash(items)
    HASH_FILE.write_text(digest + "\n")
    print(f"  wrote {len(items)} items -> {TESTSET.name}")
    print(f"  content hash: {digest[:16]}...  -> {HASH_FILE.name}")


def load(strict: bool = True) -> list[dict[str, Any]]:
    """Load the frozen set, refusing to proceed if it no longer matches its hash.

    This is the whole reason the hash exists. A gate that silently accepts a
    changed testset reports score movements that mix "the model changed" with
    "the test changed", and no amount of dashboarding can separate them again.
    """
    if not TESTSET.exists():
        raise SystemExit("no testset — run: python -m evals.build_testset")
    with open(TESTSET) as fh:
        items = [json.loads(line) for line in fh if line.strip()]
    if strict and HASH_FILE.exists():
        expected = HASH_FILE.read_text().strip()
        actual = content_hash(items)
        if actual != expected:
            raise SystemExit(
                f"testset_v1.jsonl has changed.\n"
                f"  expected {expected[:16]}...\n  found    {actual[:16]}...\n"
                "Scores from a modified testset are not comparable with earlier runs. "
                "Either restore the file, or rebuild it AND bump the version so the two "
                "sets stay distinguishable."
            )
    return items


def stats(items: list[dict[str, Any]]) -> None:
    """Describe the frozen set."""
    print(f"\n  {len(items)} items")
    for field in ("lang", "source"):
        counts = Counter(i[field] for i in items)
        print(f"    by {field}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    refusals = sum(1 for i in items if i["expects_refusal"])
    print(f"    expecting a refusal: {refusals}")
    print(
        f"    with a reference (context recall can run): "
        f"{sum(1 for i in items if i['reference'])}\n"
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="python -m evals.build_testset", description=__doc__)
    parser.add_argument("--stats", action="store_true", help="describe the frozen file")
    args = parser.parse_args()
    if args.stats:
        stats(load(strict=False))
        return
    items = build()
    write(items)
    stats(items)


if __name__ == "__main__":
    main()
