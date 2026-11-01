"""Carga de credenciales desde .env. Nunca imprime secretos."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    load_dotenv(ROOT / ".env", override=False)


def get(name: str, required: bool = False) -> str | None:
    load_env()
    val = os.getenv(name) or None
    if required and not val:
        raise RuntimeError(f"Falta la variable {name} en .env")
    return val

