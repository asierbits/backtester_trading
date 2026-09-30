"""Descubrimiento automático de módulos.

Cada carpeta dentro de `modulos/` que tenga un archivo `modulo.py` con:
    NOMBRE = "Mi módulo"
    ORDEN = 20
    def paginas() -> list[st.Page]: ...
aparece sola en el menú lateral. Añadir "voz" o "prospección" no obliga a tocar
nada del módulo de trading ni de app.py.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

CARPETA = Path(__file__).resolve().parent


def descubrir() -> list[ModuleType]:
    """Módulos disponibles, ordenados por ORDEN."""
    encontrados = []
    for carpeta in sorted(CARPETA.iterdir()):
        if (carpeta / "modulo.py").exists() and not carpeta.name.startswith("_"):
            encontrados.append(importlib.import_module(f"modulos.{carpeta.name}.modulo"))
    return sorted(encontrados, key=lambda m: getattr(m, "ORDEN", 100))
