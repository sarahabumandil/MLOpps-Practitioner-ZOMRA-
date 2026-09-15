"""All paths, hyperparameters and ports live here. No hardcoded paths elsewhere."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARASENT_", env_file=".env", extra="ignore")

    # Paths
    project_root: Path = Path(__file__).resolve().parents[2]
    raw_data_path: Path = project_root / "data" / "raw" / "arabic_reviews.csv"
    model_dir: Path = project_root / "models"
    baseline_model_path: Path = model_dir / "baseline_tfidf_lr.joblib"
    onnx_model_path: Path = model_dir / "model.onnx"

    # AraBERT fine-tune (used by notebooks/finetune_arabert_colab.ipynb — GPU runtime)
    transformer_model_name: str = "aubmindlab/bert-base-arabertv02"
    transformer_output_dir: Path = model_dir / "arabert-sentiment"
    max_seq_length: int = 128
    num_labels: int = 3  # negative, neutral, positive

    # Training (baseline)
    random_seed: int = 42
    test_size: float = 0.2

    # MLflow
    mlflow_tracking_uri: str = f"sqlite:///{project_root}/mlflow.db"
    mlflow_experiment_name: str = "arasent-sentiment"
    mlflow_registry_model_name: str = "ArabicSentiment"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Drift monitoring
    psi_alert_threshold: float = 0.25

    # Logging
    log_level: str = "INFO"


settings = Settings()
LABELS = ["negative", "neutral", "positive"]
