# Session 4 — one command per thing you do in the lab.
#
#   make up          Prometheus + Grafana          (2 containers)
#   make up-langfuse + the Langfuse stack          (6 more, profile: langfuse)
#   make bootstrap   fit + freeze the model        (once)
#   make api         the instrumented API on :8001 (Prometheus scrapes this)
#   make traffic     ~20 rps of rides              (another terminal)
#   make incident-01 inject one incident           (01..15, see list-incidents)
#   make heal        clear every active incident
#
# Ports come from .env.example — Prometheus is on 9095, not 9090.

SHELL := /bin/bash
# `python` is not on PATH on a stock macOS (and pyenv shims can shadow it),
# so default to python3 and let a virtualenv override it:
#     make api PY=.venv/bin/python
PY ?= python3
COMPOSE ?= docker compose

.DEFAULT_GOAL := help
.PHONY: help up up-langfuse down bootstrap api traffic seed doctor test heal \
        list-incidents drift drift-loop gate replay judge calibrate testset ragas \
        ragas-gate cost concept-drift

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'
	@echo "  incident-NN      inject incident NN (01..15); make list-incidents"
	@echo "  reveal-NN        instructor only: cause + signals for NN"

up: ## Start Prometheus + Grafana
	$(COMPOSE) up -d

up-langfuse: ## Start Prometheus + Grafana + the Langfuse stack
	$(COMPOSE) --profile langfuse up -d

down: ## Stop the stack, keeping volumes
	$(COMPOSE) down

bootstrap: ## Fit the model and freeze the drift reference (run once)
	$(PY) -m services.ride_model

api: ## Run the instrumented ride-duration API on :8001
	$(PY) -m uvicorn services.model_api:app --host 0.0.0.0 --port 8001

traffic: ## Drive the API at ~20 rps until Ctrl-C
	$(PY) -m tools.traffic

drift: ## Run one drift window against the frozen reference
	$(PY) -m jobs.daily_drift

drift-loop: ## Run the drift job forever, one simulated day per minute
	$(PY) -m jobs.daily_drift --loop 60

concept-drift: ## Page-Hinkley + ADWIN over the late-label error stream
	$(PY) -m jobs.concept_drift --by-city

gate: ## Ask the retrain gate for a decision, with its reasons
	$(PY) -m jobs.retrain_gate

replay: ## Re-score logged serving vectors offline and compare
	$(PY) -m tools.replay

judge: ## Run the LLM judge on its two shipped samples (one Arabic, one English)
	$(PY) -m evals.judge --demo

calibrate: ## Cohen's kappa vs the labelled set, reported per language
	$(PY) -m evals.calibrate_judge

testset: ## Rebuild the frozen 60-item evaluation set and its hash
	$(PY) -m evals.build_testset

ragas: ## RAGAS on the frozen testset (add ARGS="--gate" for the CI gate)
	$(PY) -m evals.run_ragas --source dataset --limit 10 $(ARGS)

ragas-gate: ## The CI quality gate — exits non-zero below threshold
	$(PY) -m evals.run_ragas --source dataset --gate

cost: ## Incident 08 — token volume and amortised cost, min/max week
	$(PY) -m tools.cost_report --weeks 4

seed: ## Backfill ~14 simulated days so no dashboard starts empty
	$(PY) -m tools.seed

doctor: ## Check services, artifacts, ports and models; print a checklist
	$(PY) -m tools.doctor

test: ## Run the metric-contract and cardinality guards
	$(PY) -m pytest tests -q

# One pattern rule covers incident-01 .. incident-15. Unimplemented ids (11, 12,
# 14) fail with the homework message rather than silently doing nothing.
incident-%: ## (see list-incidents)
	$(PY) -m incidents.inject --incident $*

reveal-%:
	$(PY) -m incidents.inject --reveal $*

heal: ## Clear every active incident
	$(PY) -m incidents.inject --heal

list-incidents: ## List injectable incident ids (no spoilers)
	$(PY) -m incidents.inject --list
