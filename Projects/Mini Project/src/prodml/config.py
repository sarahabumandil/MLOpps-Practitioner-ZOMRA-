"""Central configuration. All paths, hyperparameters and ports live here —
never hardcoded elsewhere. Values are overridable via environment variables.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PRODML_", env_file=".env", extra="ignore")

    # Paths
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    model_path: Path = models_dir / "model.pkl"
    onnx_model_path: Path = models_dir / "model.onnx"

    # Training
    random_seed: int = 42
    val_size: float = 0.2
    n_estimators: int = 200
    max_depth: int = 12

    # Serving
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_version: str = "0.1.0"

    # Logging
    log_level: str = "INFO"


settings = Settings()
