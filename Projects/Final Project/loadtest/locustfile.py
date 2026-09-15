"""Load test — Module 3 requirement: realistic payload mix, not the same row 10,000
times (caching would lie about throughput). Point --host at the FastAPI baseline
(default) or the BentoML service to compare them directly.

Run:  locust -f loadtest/locustfile.py --host http://localhost:8000
Headless, 100 users, 2 min:
  locust -f loadtest/locustfile.py --host http://localhost:8000 \
    --users 100 --spawn-rate 10 --run-time 2m --headless --csv=reports/locust_fastapi
"""
from __future__ import annotations

import random

from locust import HttpUser, between, task

POSITIVE_REVIEWS = [
    "المنتج رائع جدا وسريع في التوصيل",
    "جودة ممتازة وأنصح بالشراء بشدة لكل من يريد منتج يدوم طويلا",
    "تجربة شراء رائعة والخدمة سريعة والتغليف ممتاز",
]
NEGATIVE_REVIEWS = [
    "المنتج سيء جدا ولا يستحق السعر المدفوع فيه إطلاقا",
    "التوصيل متأخر جدا والخدمة سيئة ولم يتم الرد على شكواي",
    "المنتج وصل مكسور ولم يتم استرجاع المال حتى الان",
]
NEUTRAL_REVIEWS = [
    "المنتج عادي لا بأس به",
    "لا سيء ولا ممتاز، متوسط الجودة والسعر مناسب نوعا ما",
]

ALL_REVIEWS = POSITIVE_REVIEWS + NEGATIVE_REVIEWS + NEUTRAL_REVIEWS


class SentimentApiUser(HttpUser):
    wait_time = between(1, 3)

    @task(80)
    def single_predict(self):
        text = random.choice(ALL_REVIEWS)
        self.client.post("/predict", json={"text": text}, name="/predict")

    @task(15)
    def batch_predict(self):
        reviews = [{"text": random.choice(ALL_REVIEWS)} for _ in range(random.randint(2, 8))]
        self.client.post("/predict/batch", json={"reviews": reviews}, name="/predict/batch")

    @task(5)
    def metadata(self):
        self.client.get("/metadata", name="/metadata")
