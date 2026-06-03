from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"

class Settings(BaseSettings):
    pass