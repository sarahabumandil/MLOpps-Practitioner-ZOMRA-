"""Do we believe the judge? Cohen's kappa against known labels, per language.

A judge produces numbers whether or not it is any good, and the numbers look
identical either way. Before quoting a single score you have to measure the
judge itself against labels you trust.

Raw agreement is not enough. On a set that is 80% grounded, a judge that answers
"grounded" every single time scores 80% agreement while measuring nothing at
all. Cohen's kappa subtracts the agreement you would get by chance from the
marginals, so that degenerate judge scores about 0.

    python -m evals.calibrate_judge              # score the shipped set
    python -m evals.calibrate_judge --rebuild    # regenerate the set, then score

**Expect Arabic kappa to come out materially below English.** That is the finding,
not a bug: slide 53's claim that judges are weaker at Arabic claim decomposition,
reproduced on the student's own laptop. Do not "fix" it by swapping in a larger
model — run it on a 7B and again on a 70B and show the gap narrowing instead.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from evals.judge import Judge, ParseFailure
from evals.knowledge_base import NOTES

CALIBRATION_SET = Path(__file__).resolve().parent / "calibration_set.jsonl"

#: Below this, the judge and the labels agree barely more than chance would
#: predict, and no score it produces should be quoted.
KAPPA_TRUST_FLOOR = 0.80

#: Claims that are plausible, often true in the real world, and NOT in the
#: context they get attached to. That distinction — true versus grounded — is
#: the single thing a groundedness rubric exists to measure, so the label set
#: has to contain it or the calibration proves nothing.
DISTRACTORS_EN = [
    "This metric was originally developed for credit risk scorecards in the 1990s.",
    "Most teams recompute this hourly in production.",
    "The threshold is defined in the ISO 24028 standard.",
    "Netflix published a paper showing this approach halved their incident count.",
    "It is generally recommended to use at least twenty bins for this.",
]
DISTRACTORS_AR = [
    "طُوِّر هذا المقياس أصلًا لبطاقات تقييم مخاطر الائتمان في التسعينيات.",
    "تعيد معظم الفرق حساب هذا كل ساعة في بيئة الإنتاج.",
    "هذا الحد معرَّف في المواصفة القياسية ISO 24028.",
    "نشرت نتفليكس ورقة بحثية تبين أن هذا النهج خفّض عدد الحوادث إلى النصف.",
    "يُنصح عمومًا باستخدام عشرين صندوقًا على الأقل لهذا الغرض.",
]


def build_set(n_per_lang: int = 25, seed: int = 7) -> list[dict[str, Any]]:
    """Construct the labelled set from the knowledge base.

    The labels are not one person's opinion — they are known by CONSTRUCTION.
    A `grounded=True` item's answer is assembled only from sentences in its own
    context; a `grounded=False` item is the same answer with one distractor
    claim spliced in. That is more auditable than fifty judgement calls nobody
    can re-check, and it means a disagreement is unambiguously the judge's.
    """
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []

    for lang, distractors in (("en", DISTRACTORS_EN), ("ar", DISTRACTORS_AR)):
        notes = [n for n in NOTES if len(n[lang][1].split(".")) >= 2]
        rng.shuffle(notes)
        for i in range(n_per_lang):
            note = notes[i % len(notes)]
            title, text = note[lang]
            sentences = [s.strip() for s in text.replace("۔", ".").split(".") if s.strip()]
            answer = ". ".join(sentences[:2]) + "."
            grounded = i % 2 == 0
            if not grounded:
                answer = f"{answer} {rng.choice(distractors)}"
            question = (
                f"What does the note on {title} say?"
                if lang == "en"
                else f"ماذا تقول الملاحظة عن {title}؟"
            )
            items.append(
                {
                    "id": f"{lang}-{note['id']}-{i}",
                    "lang": lang,
                    "question": question,
                    "answer": answer,
                    "context": text,
                    "grounded": grounded,  # the human label
                }
            )
    return items


def write_set(items: list[dict[str, Any]], path: Path = CALIBRATION_SET) -> None:
    """Freeze the set to JSONL."""
    with open(path, "w") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  wrote {len(items)} labelled items -> {path.name}")


def load_set(path: Path = CALIBRATION_SET) -> list[dict[str, Any]]:
    """Read the frozen set, building it first if it is missing."""
    if not path.exists():
        write_set(build_set(), path)
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> tuple[float, float]:
    """Return (raw agreement, Cohen's kappa) for a list of (human, judge) pairs.

    kappa = (po - pe) / (1 - pe), where pe is the agreement expected from the two
    raters' marginal rates alone. It is the correction that stops "always say
    grounded" from looking like a competent judge on an 80%-grounded set.
    """
    n = len(pairs)
    if not n:
        return float("nan"), float("nan")
    po = sum(1 for h, j in pairs if h == j) / n
    h_true = sum(1 for h, _ in pairs if h) / n
    j_true = sum(1 for _, j in pairs if j) / n
    pe = h_true * j_true + (1 - h_true) * (1 - j_true)
    if pe == 1.0:  # both raters unanimous — kappa is undefined, not perfect
        return po, float("nan")
    return po, (po - pe) / (1 - pe)


def calibrate(items: list[dict[str, Any]], judge: Judge) -> dict[str, Any]:
    """Score every item and compare the judge's verdict with the label."""
    by_lang: dict[str, list[tuple[bool, bool]]] = {}
    failures: dict[str, int] = {}
    disagreements: list[dict[str, Any]] = []

    for item in items:
        lang = item["lang"]
        by_lang.setdefault(lang, [])
        failures.setdefault(lang, 0)
        try:
            result = judge.groundedness(item["question"], item["answer"], item["context"])
        except ParseFailure:
            failures[lang] += 1
            continue
        # Binarise: "grounded" means every atomic claim was supported.
        judged = result["value"] >= 0.999
        by_lang[lang].append((item["grounded"], judged))
        if judged != item["grounded"]:
            disagreements.append(
                {
                    "id": item["id"],
                    "lang": lang,
                    "human": item["grounded"],
                    "judge": judged,
                    "score": round(result["value"], 3),
                    "unsupported": result["unsupported"][:1],
                }
            )
    return {"by_lang": by_lang, "failures": failures, "disagreements": disagreements}


def report(outcome: dict[str, Any], fingerprint: dict[str, str]) -> bool:
    """Print per-language kappa and say plainly whether the judge is usable."""
    print("\n  judge:", " ".join(f"{k}={v}" for k, v in fingerprint.items()))
    print("\n  agreement with the labelled set, PER LANGUAGE:\n")
    print(f"    {'lang':<6} {'n':>4} {'raw':>7} {'kappa':>7}  {'parse-fail':>10}   verdict")

    trusted = True
    kappas: dict[str, float] = {}
    for lang, pairs in sorted(outcome["by_lang"].items()):
        po, kappa = cohens_kappa(pairs)
        kappas[lang] = kappa
        failed = outcome["failures"].get(lang, 0)
        ok = kappa == kappa and kappa >= KAPPA_TRUST_FLOOR  # NaN-safe
        verdict = "usable" if ok else "DO NOT TRUST"
        trusted &= ok
        print(f"    {lang:<6} {len(pairs):>4} {po:>7.3f} {kappa:>7.3f}  {failed:>10}   {verdict}")

    if len(kappas) == 2 and all(k == k for k in kappas.values()):
        gap = kappas.get("en", 0) - kappas.get("ar", 0)
        print(f"\n    English minus Arabic kappa: {gap:+.3f}")
        if gap > 0.05:
            print(
                "    The judge is measurably worse at Arabic. This is the expected result\n"
                "    and it is the finding, not a bug — a small model decomposes Arabic\n"
                "    claims less reliably than English ones. An Arabic score and an English\n"
                "    score from this judge are NOT on the same scale and must never be\n"
                "    averaged together or compared to one threshold."
            )

    if not trusted:
        print(
            f"\n  REFUSING TO CERTIFY. Kappa below {KAPPA_TRUST_FLOOR:.2f} means the judge and\n"
            "  the labels agree barely more than chance would predict. Every score this\n"
            "  judge produces is uninterpretable until that is fixed: raise num_ctx, sharpen\n"
            "  the rubric, or use a bigger model — and re-run this before quoting anything.\n"
        )

    worst = outcome["disagreements"][:5]
    if worst:
        print("  where they disagreed:")
        for d in worst:
            direction = "judge too harsh" if d["human"] and not d["judge"] else "judge too lenient"
            print(f"    [{d['lang']}] {d['id'][:28]:<28} score={d['score']:.2f}  {direction}")
    return trusted


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="python -m evals.calibrate_judge", description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="regenerate the labelled set")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N items")
    args = parser.parse_args()

    if args.rebuild:
        write_set(build_set())
    items = load_set()
    if args.limit:
        items = (
            items[: args.limit // 2] + items[len(items) // 2 : len(items) // 2 + args.limit // 2]
        )
    print(
        f"  calibrating on {len(items)} labelled items "
        f"({sum(1 for i in items if i['lang'] == 'ar')} Arabic)"
    )

    judge = Judge()
    outcome = calibrate(items, judge)
    raise SystemExit(0 if report(outcome, judge.config.fingerprint()) else 1)


if __name__ == "__main__":
    main()
