"""Every incident flag must be read by something. Guards against an inert lab.

This test exists because incident 07 shipped broken: `inject.py` wrote
`PROMPT_PRODUCTION_LABEL` into the state file, the terminal printed a convincing
symptom card, the deploy annotation appeared on the dashboard — and nothing in
the codebase ever read the flag. The system was completely healthy. Students
would have hunted a bug that was not there, which is the worst possible outcome
for a diagnostic exercise.

Nothing failed, which is exactly why it survived a manual test: an injector that
does nothing looks identical to one that works until you check the signal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from incidents.catalog import CATALOG

PROJECT_ROOT = Path(__file__).resolve().parents[1]
#: Where a flag may legitimately be read. catalog.py is excluded: declaring a
#: flag is not reading it, and that confusion is what let 07 slip through.
SEARCH_DIRS = ("services", "tools", "jobs", "evals", "incidents")


def sources() -> str:
    """Every Python source file that could act on a flag, concatenated."""
    text = []
    for directory in SEARCH_DIRS:
        for path in (PROJECT_ROOT / directory).rglob("*.py"):
            if path.name == "catalog.py":
                continue
            text.append(path.read_text())
    text.append((PROJECT_ROOT / "ollama_langfuse_rag.py").read_text())
    return "\n".join(text)


ALL_FLAGS = sorted({flag for inc in CATALOG.values() for flag in inc.flags})


@pytest.mark.parametrize("flag", ALL_FLAGS)
def test_flag_has_a_reader(flag: str) -> None:
    """A flag nobody reads is an incident that injects nothing."""
    assert flag in sources(), (
        f"{flag} is declared in the catalogue but no code reads it. The injector "
        "would print a symptom, write a deploy row, and change nothing — the worst "
        "possible outcome for a diagnostic exercise."
    )


def test_every_implemented_incident_does_something() -> None:
    """An implemented incident either sets a flag or is a measurement."""
    inert = [
        inc.id
        for inc in CATALOG.values()
        if inc.implemented and not inc.flags and inc.kind != "measurement"
    ]
    assert not inert, f"implemented incidents with no effect: {inert}"


def test_symptom_cards_never_name_the_cause() -> None:
    """The symptom is what a user reports. Naming the mechanism ruins the lab."""
    spoilers = ("scaler", "cardinality", "miles", "normalis", "label was moved", "because the")
    leaked = {
        inc.id: word
        for inc in CATALOG.values()
        if inc.implemented
        for word in spoilers
        if word in inc.symptom.lower()
    }
    assert not leaked, f"symptom cards leak their cause: {leaked}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
