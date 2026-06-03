from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
CACHE_MODEL_PATH: Path = ROOT_DIR / ".cache"

class Settings(BaseSettings):
    embedding_model_name: str
    llm_name: str

    embedding_batch_size: int
    device: str = "cpu"
