"""Every alert's runbook link must point at a heading that exists.

An alert annotation is read exactly once — at 3am, by someone who did not write
it. A link to a heading that was renamed or never written is worse than no link,
because it costs the reader time before it tells them nothing.

The anchor rules match GitHub's: lowercase, spaces to hyphens, everything else
that is not alphanumeric or a hyphen dropped.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULES = PROJECT_ROOT / "monitoring" / "prometheus" / "alert_rules.yml"
RUNBOOK = PROJECT_ROOT / "RUNBOOK.md"


def github_anchor(heading: str) -> str:
    """Convert a markdown heading to the anchor GitHub generates for it."""
    text = heading.strip().lstrip("#").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", "-", text)


def runbook_anchors() -> set[str]:
    """Every anchor RUNBOOK.md defines."""
    return {
        github_anchor(line) for line in RUNBOOK.read_text().splitlines() if line.startswith("#")
    }


def alert_runbook_links() -> dict[str, str]:
    """Map alert name -> the anchor its runbook annotation points at."""
    rules = yaml.safe_load(RULES.read_text())
    links = {}
    for group in rules["groups"]:
        for rule in group["rules"]:
            link = rule.get("annotations", {}).get("runbook", "")
            links[rule["alert"]] = link.partition("#")[2]
    return links


def test_every_alert_has_a_runbook_annotation() -> None:
    """An alert with no runbook is a pager with no instructions attached."""
    missing = [name for name, anchor in alert_runbook_links().items() if not anchor]
    assert not missing, f"alerts with no runbook link: {missing}"


def test_every_runbook_link_resolves() -> None:
    """The anchor each alert points at must actually exist in RUNBOOK.md."""
    anchors = runbook_anchors()
    broken = {
        name: anchor for name, anchor in alert_runbook_links().items() if anchor not in anchors
    }
    assert not broken, (
        f"alerts link to headings that do not exist in RUNBOOK.md: {broken}. "
        f"Available anchors: {sorted(anchors)}"
    )


def test_internal_runbook_links_resolve() -> None:
    """RUNBOOK.md's own cross-references must resolve too."""
    anchors = runbook_anchors()
    internal = set(re.findall(r"\]\(#([a-z0-9-]+)\)", RUNBOOK.read_text()))
    broken = internal - anchors
    assert not broken, f"dead internal links in RUNBOOK.md: {sorted(broken)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
