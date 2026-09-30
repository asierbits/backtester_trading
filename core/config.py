"""Carga de la configuración (config.yaml) y de los secretos (.env)."""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CONFIG = RAIZ / "config.yaml"


@lru_cache(maxsize=1)
def _config_en_disco() -> dict[str, Any]:
    """Lee config.yaml una sola vez por proceso."""
    with open(RUTA_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_config() -> dict[str, Any]:
    """Devuelve una copia de la configuración para poder modificarla sin efectos secundarios."""
    return copy.deepcopy(_config_en_disco())


def guardar_config(config: dict[str, Any]) -> None:
    """Escribe la configuración en config.yaml (usado por la página de Ajustes)."""
    with open(RUTA_CONFIG, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    _config_en_disco.cache_clear()


def ruta_datos() -> Path:
    """Carpeta donde viven los Parquet y la base de datos."""
    ruta = RAIZ / cargar_config()["general"]["ruta_datos"]
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def ruta_informes() -> Path:
    """Carpeta de informes exportados."""
    ruta = RAIZ / cargar_config()["general"]["ruta_informes"]
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def clave_anthropic() -> str | None:
    """Lee ANTHROPIC_API_KEY de .env. Nunca se registra en logs ni se guarda en la base de datos."""
    load_dotenv(RAIZ / ".env")
    clave = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not clave or clave == "pon_tu_clave_aqui":
        return None
    return clave
