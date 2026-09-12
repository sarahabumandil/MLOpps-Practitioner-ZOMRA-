# The MLOps Practitioner — Course Notes

> My personal documentation repo for **The MLOps Practitioner** course, run by **MLOps MENA Community**.
> Cohort 1 · Aug 15 → Oct 2, 2026 · 7 weeks · 5 live lessons · Free

![Status](https://img.shields.io/badge/status-running-brightgreen) ![Cohort](https://img.shields.io/badge/cohort-1-blue) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 📖 About the Course

By the end of this course, students take a machine learning model from notebook to production — building automated CI/CD pipelines, experiment tracking, and scheduled retraining. Students serve predictions at scale using **FastAPI, BentoML, Triton, and vLLM**, monitor for drift before users notice, and optimize models for GPU, CPU, and edge devices.

| | |
|---|---|
| **Lessons** | 5 live lessons |
| **Duration** | 7 weeks |
| **Dates** | Aug 15 → Oct 2, 2026 |
| **Price** | Free |
| **Rating** | ⭐ 4.9 (7) |
| **Delivered with** | Zomra (educational partner) |
| **Instructor** | Aya Nasser Salama — Founder of MLOps MENA & Senior MLOps Engineer |

---

## 🎯 The 7 Objectives

1. Structure ML projects professionally using Python packaging, OOP, type hints, and build production-grade REST APIs with **FastAPI**/**Litestar** — containerized with Docker, tested with pytest.
2. Track experiments, version data, and manage model lifecycle using **MLflow** and **DVC** — automate the train → test → build → push pipeline with **GitHub Actions** and **Terraform**.
3. Implement Continuous Training pipelines that retrain, evaluate, and promote models automatically when data drifts or performance degrades — no human intervention.
4. Choose the right inference pattern and serve models with the full stack: **FastAPI → BentoML → TensorRT/Triton (GPU) → ONNX Runtime/OpenVINO (CPU) → vLLM (LLMs)**.
5. Release models safely using canary rollouts, A/B testing, blue/green deployments, and shadow mode — with automatic rollback.
6. Detect data/concept/label/embedding drift using PSI, KS test, Page-Hinkley, and MMD — monitor with **Prometheus, Grafana, Langfuse, RAGAS**.
7. Optimize trained models using pruning, quantization (PTQ/QAT), knowledge distillation, TensorRT, OpenVINO, and TFLite — measuring the accuracy/latency/size tradeoff.

## 🛠️ Tools Used in This Course

`FastAPI` `Litestar` `Docker` `pytest` `MLflow` `DVC` `GitHub Actions` `Terraform` `Apache Airflow` `BentoML` `Triton` `vLLM` `TensorRT` `ONNX Runtime` `OpenVINO` `TFLite` `Evidently AI` `Prometheus` `Grafana` `Langfuse` `RAGAS`

**Prerequisites:** basic Python (functions, classes, pandas, scikit-learn) · trained at least one ML model before · a laptop with Docker and ≥8GB RAM.

---

## 🗓️ Weeks

<details open>
<summary><b>Week 1 · Aug 15 – Aug 21 — From Notebook to Production-Ready Code</b></summary>

The MLOps Maturity Model · Python Packaging & Project Structure · Building ML APIs (FastAPI vs Litestar) · Serialization Formats · Docker & Containerization · Structured Logging · Testing ML Code with pytest

**Module project** — A fully containerized ML API with a test suite, structured logs, and a 3-command README.
> 🎥 Recording available
</details>

<details>
<summary><b>Week 2 · Aug 22 – Aug 28 — MLOps Core: Experiment Tracking, Versioning & Automation</b></summary>

MLflow Experiment Tracking · MLflow Model Registry · Continuous Training with MLflow · Data Versioning with DVC · CI/CD with GitHub Actions · Infrastructure as Code with Terraform

**Module project** — A fully automated pipeline triggered by GitHub Actions: trains, evaluates against production, promotes only if metrics improve, builds a Docker image. Every run in MLflow, every dataset version in DVC.
> 🎥 Recording available
</details>

<details>
<summary><b>Week 3 · Aug 29 – Sep 4 — 1st Half of the Project (Implementation + Revise)</b></summary>

No lecture. Work on the chosen project and apply the principles from the first two lectures.
</details>

<details>
<summary><b>Week 4 · Sep 5 – Sep 11 — Inference, Serving & Release Strategies</b></summary>

Orchestration with Apache Airflow · Why Inference Patterns Matter · Three Inference Patterns · What is Model Serving · CAT 1 FastAPI · CAT 2 BentoML · CAT 3 TensorRT + Triton · CAT 4 ONNX Runtime + OpenVINO · CAT 5 vLLM · Load Testing with Locust · Release Strategies

**Module project** — Serve the ride-duration model three ways, load test to 100 concurrent users, document the bottleneck, deploy a new version via canary rollout with automatic rollback.
> 🎥 Recording available
</details>

<details>
<summary><b>Week 5 · Sep 12 – Sep 18 — Model Optimization: Faster, Smaller, Cheaper (🔜 next session)</b></summary>

Why Optimization Matters · Pruning · Post-Training Quantization · Quantization-Aware Training · Knowledge Distillation · TensorRT · ONNX Runtime · OpenVINO · Edge Deployment · LLM-Specific Quantization (AWQ, GPTQ) · Benchmarking and the Optimization Decision

> 🗓️ Sunday 13 Sept, 7:00 pm Cairo
</details>

<details>
<summary><b>Week 6 · Sep 19 – Sep 25 — Observability & Drift Detection</b></summary>

Why Production Models Degrade · Drift Taxonomy · Statistical Detection Methods · Evidently AI · Page-Hinkley and ADWIN · Label and Prediction Drift · Embedding Drift · Prometheus + Grafana · Langfuse · RAGAS · Cost and Token Monitoring · Guardrails

> 🗓️ Sunday 20 Sept, 7:00 pm Cairo
</details>

<details>
<summary><b>Week 7 · Sep 26 – Oct 2 — Final Project</b></summary>

Ship the project end to end, present it to the community, and get the repo reviewed.
</details>

---

## 👥 Study Groups (pick by experience, not job title)

| Group | Name | For |
|---|---|---|
| 1 | MLOps Beginner | Junior in ML/DS, comfortable with Python & ML fundamentals, new to Linux/Git/Docker |
| 2 | MLOps Intermediate | Backend/software engineering background, working with FastAPI/Flask, solid with Git & Docker |
| 3 | Advanced MLOps / Cloud | Hands-on cloud (AWS/GCP/Azure), Kubernetes, Terraform, monitoring experience |
| 4 | DevOps → MLOps | Currently DevOps/Platform/Cloud/SRE, strong ops skills, needs the ML half |

---

## 👤 The Team

<table>
<tr>
<td align="center" width="200">
<img src="images/team/aya-nasser-salama.jpg" width="120" style="border-radius:50%"><br>
<b>Aya Nasser Salama</b><br>
<sub>Founder</sub>
</td>
<td align="center" width="200">
<img src="images/team/basem-abusaif.jpg" width="120" style="border-radius:50%"><br>
<b>Basem Abusaif</b><br>
<sub>Community Director</sub>
</td>
<td align="center" width="200">
<img src="images/team/omar-salah.jpg" width="120" style="border-radius:50%"><br>
<b>Omar Salah</b><br>
<sub>Community Director</sub>
</td>
</tr>
<tr>
<td align="center">
<img src="images/team/mohamed-samy.jpg" width="120" style="border-radius:50%"><br>
<b>Mohamed Samy Mansour</b><br>
<sub>AI Instructor Lead</sub>
</td>
<td align="center">
<b>ZA</b><br>
<b>Zakaria Ahmed</b><br>
<sub>AI Content Lead</sub>
</td>
<td align="center">
<img src="images/team/radwa-khattab.jpg" width="120" style="border-radius:50%"><br>
<b>Radwa Khattab</b><br>
<sub>AI Community Growth & Partnerships Lead</sub>
</td>
</tr>
<tr>
<td align="center">
<img src="images/team/mariam-qotob.jpg" width="120" style="border-radius:50%"><br>
<b>Mariam Qotob</b><br>
<sub>AI Sessions & Mentorship Lead</sub>
</td>
<td align="center">
<img src="images/team/mahmoud-abu-alnour.jpg" width="120" style="border-radius:50%"><br>
<b>Mahmoud Abu Al-Nour</b><br>
<sub>AI Platform & Social Media Lead</sub>
</td>
<td align="center">
<b>?</b><br>
<b>Khadija Ahaidous</b><br>
<sub>AI Research Lead</sub>
</td>
</tr>
</table>

### Bios

- **Aya Nasser Salama** — Founder · Senior MLOps & LLMOps Engineer with 6+ years in AI. At Unifonic she is the central MLOps/LLMOps support across AI teams — agentic products with LangGraph and MCP, LLM evaluation and observability, and 30B+ model serving on Kubernetes. Designed and teaches "Production ML Engineering" at ITI, and built Valeo's first production RAG system. · [LinkedIn](#) · `aya@mlopsmena.com`
- **Basem Abusaif** — Community Director · AI Engineer working on 3D perception and generative AI, 5+ years building deep learning pipelines for autonomous systems. Leads the 3D perception stack at Wakeb Data, previously productionised LiDAR simulation models at Valeo. Completing an MSc in Informatics at Nile University. · `basem@mlopsmena.com`
- **Omar Salah** — Community Director · Co-runs the community day to day — sessions, study groups, and keeping the programme moving. · `omar@mlopsmena.com`
- **Mohamed Samy Mansour** — AI Instructor Lead · AI and Data Science engineer, Computer Engineering background from Mansoura University. Mid-Level Data Scientist at Andalusia Healthcare Group, previously AI & Data Science Engineer at Etisalat Misr, working across ML pipelines, model serving, and LLM/RAG applications. · `mohamedsamy@mlopsmena.com`
- **Zakaria Ahmed** — AI Content Lead · Owns the community's written content — roadmaps, articles, and the material that goes out with every session. · `zakaria@mlopsmena.com`
- **Khadija Ahaidous** — AI Research Lead · Leads the community's research direction — the papers read, the work published, and the academic mentoring that goes with them. · `khadijaahaidous@mlopsmena.com`
- **Radwa Khattab** — AI Community Growth & Partnerships Lead · Senior AI Engineer with 6+ years of experience, including two years at Microsoft as an Applied & Data Scientist, building production AI/ML systems across LLMs, agentic AI, and cloud infrastructure. Pursuing a Master's in AI at Cairo University; has taught AI/ML at two universities and Udacity. · `radwa@mlopsmena.com`
- **Mariam Qotob** — AI Sessions & Mentorship Lead · AI/ML engineer with a professional master's in AI from Queen's University, working across the full ML pipeline with applied experience in LLMs and RAG. Currently a Teaching Assistant for the Digilians Initiative. · `mariam@mlopsmena.com`
- **Mahmoud Abu Al-Nour** — AI Platform & Social Media Lead · Computer Science student focused on AI, Machine Learning, and Data Science. Builds practical AI solutions with Python, SQL, FastAPI, Docker, and MLOps tooling. · `mahmoud@mlopsmena.com`

---

## 📚 Resources

- **Course repository** — all code, notebooks, and module projects
- **All session slides** — Google Drive folder with slides for all five sessions
- **Session 1 slides** — direct link to the first session deck
- **Mini projects & final project** — link coming soon
- **Session recordings** — all five live lessons on YouTube (Session 1 stays up permanently; the rest are taken down 48h after each session)

## 🌍 Community Links

| Channel | |
|---|---|
| WhatsApp | 3,000+ members |
| LinkedIn | 3,000+ followers |
| YouTube | All recordings |
| Discord | Chat & help |
| X | Follow for updates |

---

## 📝 My Self-Study Log

> _(Personal section — update as I go through each week)_

- [x] Week 1 — From Notebook to Production-Ready Code
- [x] Week 2 — MLOps Core
- [ ] Week 3 — Project (1st half)
- [x] Week 4 — Inference, Serving & Release Strategies
- [ ] Week 5 — Model Optimization *(upcoming: Sun 13 Sept, 7pm Cairo)*
- [ ] Week 6 — Observability & Drift Detection
- [ ] Week 7 — Final Project

## ❓ FAQ (from the course page)

**Is it normal to feel lost as a junior/student, hearing a lot of unfamiliar terms?**
Yes — it means the session is doing its job. Feeling flooded is a side effect of stepping out of the notebook and discovering how much bigger the field is.

**How do I deal with new terminology?**
Write down every new term. Search it on YouTube, read the official docs, and bring what's still unclear to the community.

**Will the recordings stay available?**
Session 1 stays permanently on YouTube. The rest are taken down 48 hours after each session.

---

## 📄 License

This repository documents my personal notes on a free, community-run course.
Course content, curriculum, and material belong to **MLOps MENA Community**.
My own notes and summaries in this repo are shared under the MIT License.

```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<sub>Built while following **The MLOps Practitioner** by MLOps MENA Community · © 2026</sub>
