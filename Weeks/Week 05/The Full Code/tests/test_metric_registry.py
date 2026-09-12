"""The dashboard and the alert rules are a contract. This test enforces it.

Renaming a metric is a silent break: Grafana draws an empty panel and Prometheus
evaluates an alert that can never fire. Both fail *quietly*, which is the worst
possible failure mode for the tooling you rely on during an incident.

So: parse every PromQL expression out of model-monitoring.json and
alert_rules.yml, extract the metric names, and assert the service actually
exports them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from prometheus_client import REGISTRY

import services.model_api  # noqa: F401  — registers the metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = PROJECT_ROOT / "monitoring" / "grafana" / "dashboards" / "model-monitoring.json"
RULES = PROJECT_ROOT / "monitoring" / "prometheus" / "alert_rules.yml"

#: Exported by Prometheus itself or by the scrape, not by our service.
NOT_OURS = {"up", "prometheus_tsdb_head_series", "scrape_samples_scraped"}
#: PromQL functions and keywords that look like metric names to a regex.
KEYWORDS = {
    "rate",
    "sum",
    "by",
    "histogram_quantile",
    "increase",
    "count",
    "avg",
    "max",
    "min",
    "topk",
    "predict_linear",
    "changes",
    "and",
    "or",
    "unless",
    "le",
    "job",
    "on",
    "group_left",
    "group_right",
    "without",
    "absent",
    "delta",
    "irate",
    "clamp_max",
}
METRIC_RE = re.compile(r"\b([a-z_][a-z0-9_]{3,})\b")
SUFFIXES = ("_bucket", "_count", "_sum")
#: Label selectors `{job="model-api"}` and groupings `by (endpoint, status)` are
#: full of identifiers that are labels, not metrics. Strip both before matching.
SELECTOR_RE = re.compile(r"\{[^}]*\}")
GROUPING_RE = re.compile(
    r"\b(?:by|without|on|ignoring|group_left|group_right)\s*\([^)]*\)", re.IGNORECASE
)


def exported_names() -> set[str]:
    """Base names the service exports, without _bucket/_count/_sum suffixes."""
    names = set()
    for name, collector in list(REGISTRY._names_to_collectors.items()):
        base = name
        for suffix in SUFFIXES:
            base = base.removesuffix(suffix)
        names.add(base)
        for metric in getattr(collector, "collect", lambda: [])():
            names.add(metric.name)
    return names


def referenced_names() -> set[str]:
    """Metric names referenced by the dashboard JSON and the alert rules."""
    text = DASHBOARD.read_text() + "\n" + RULES.read_text()
    exprs = [
        t["expr"]
        for p in json.loads(DASHBOARD.read_text()).get("panels", [])
        for t in (p.get("targets") or [])
        if t.get("expr")
    ]
    exprs += re.findall(r"expr:\s*(?:>-)?\s*(.+)", RULES.read_text())
    found = set()
    for expr in exprs:
        expr = GROUPING_RE.sub(" ", SELECTOR_RE.sub(" ", expr))
        for token in METRIC_RE.findall(expr):
            if token in KEYWORDS or token in NOT_OURS:
                continue
            for suffix in SUFFIXES:
                token = token.removesuffix(suffix)
            found.add(token)
    assert text  # the files exist and are non-empty
    return found


def test_every_referenced_metric_is_exported() -> None:
    """No panel or alert may reference a metric the service does not export."""
    missing = referenced_names() - exported_names()
    assert not missing, (
        f"dashboard/alerts reference metrics nothing exports: {sorted(missing)}. "
        "A renamed metric shows up as an empty panel and an alert that never fires."
    )


def test_the_regression_contract_is_intact() -> None:
    """Session 4 monitors a REGRESSOR. Classifier metrics must not appear."""
    exported = exported_names()
    assert "model_prediction_duration_min" in exported, "the output histogram is the contract"
    for classifier_metric in ("ml_prediction_score", "positive_rate", "ml_predictions_total"):
        assert classifier_metric not in exported, (
            f"{classifier_metric} is a classification metric. This model predicts minutes; "
            "its output distribution is a histogram of minutes and its error is MAE."
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
