from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARASENT_", env_file=".env", extra="ignore")
    project_root: Path = Path(__file__).resolve().parents[2]
    raw_data_path: Path = project_root / "data" / "raw" / "arabic_reviews.csv"
    model_dir: Path = project_root / "models"
    kaggle_dataset: str = "abedkhooli/arabic-100k-reviews"
    kaggle_file: str = "ar_reviews_100k.tsv"
    auto_download: bool = False  
    max_rows: int | None = None  
    baseline_model_path: Path = model_dir / "baseline_tfidf_lr.joblib"
    onnx_model_path: Path = model_dir / "model.onnx"
    transformer_model_name: str = "aubmindlab/bert-base-arabertv02"
    transformer_output_dir: Path = model_dir / "arabert-sentiment"
    max_seq_length: int = 128
    num_labels: int = 3  
    random_seed: int = 42
    test_size: float = 0.2
    mlflow_tracking_uri: str = f"sqlite:///{project_root}/mlflow.db"
    mlflow_experiment_name: str = "arasent-sentiment"
    mlflow_registry_model_name: str = "ArabicSentiment"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    psi_alert_threshold: float = 0.25
    log_level: str = "INFO"
settings = Settings()
LABELS = ["negative", "neutral", "positive"]
