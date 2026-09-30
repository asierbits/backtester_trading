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
RUTA_CONFIG = Path(os.getenv("LABORATORIO_CONFIG", RAIZ / "config.yaml"))
RUTA_ENV = RAIZ / ".env"


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


def fusionar(base: dict[str, Any], cambios: dict[str, Any]) -> dict[str, Any]:
    """Fusiona recursivamente 'cambios' sobre una copia de 'base'."""
    resultado = copy.deepcopy(base)
    for clave, valor in cambios.items():
        if isinstance(valor, dict) and isinstance(resultado.get(clave), dict):
            resultado[clave] = fusionar(resultado[clave], valor)
        else:
            resultado[clave] = copy.deepcopy(valor)
    return resultado


def ruta_datos() -> Path:
    """Carpeta donde viven los Parquet y la base de datos.

    La variable de entorno LABORATORIO_DATOS permite cambiarla (lo usan los tests).
    """
    ruta = Path(os.getenv("LABORATORIO_DATOS", RAIZ / cargar_config()["general"]["ruta_datos"]))
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def ruta_informes() -> Path:
    """Carpeta de informes exportados."""
    ruta = Path(os.getenv("LABORATORIO_INFORMES", RAIZ / cargar_config()["general"]["ruta_informes"]))
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def leer_env(nombre: str) -> str | None:
    """Lee una variable de .env (o del entorno). Devuelve None si está vacía."""
    load_dotenv(RUTA_ENV)
    valor = os.getenv(nombre, "").strip()
    return valor or None


def clave_anthropic() -> str | None:
    """Lee ANTHROPIC_API_KEY de .env. Nunca se registra en logs ni se guarda en la base de datos."""
    clave = leer_env("ANTHROPIC_API_KEY")
    if not clave or clave == "pon_tu_clave_aqui":
        return None
    return clave


def guardar_en_env(nombre: str, valor: str) -> None:
    """Escribe o sustituye una variable en .env (el archivo está en .gitignore)."""
    lineas: list[str] = []
    if RUTA_ENV.exists():
        lineas = RUTA_ENV.read_text(encoding="utf-8").splitlines()
    nuevas = [linea for linea in lineas if not linea.startswith(f"{nombre}=")]
    nuevas.append(f"{nombre}={valor.strip()}")
    RUTA_ENV.write_text("\n".join(nuevas) + "\n", encoding="utf-8")
    os.environ[nombre] = valor.strip()
