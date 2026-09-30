"""Clase base de las estrategias y registro de familias.

Una estrategia convierte velas en una serie de **posiciones objetivo** decididas al
cierre de cada vela: 1 = largo, 0 = fuera de mercado, -1 = corto (solo si se permiten).
El motor de backtest ejecuta la posición decidida en t en la vela t+1, nunca antes.

Para añadir una estrategia nueva:
1. Crea un archivo en esta carpeta con una clase que herede de `Estrategia`.
2. Decórala con `@registrar`, define `familia`, `nombre_legible` y `_posiciones`.
3. Añade su espacio de parámetros en `config.yaml` → `parametros.<familia>`.
4. Impórtala en `estrategias/__init__.py`.
"""

from __future__ import annotations

import hashlib
import itertools
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from modulos.trading.estrategias.indicadores import CacheIndicadores

FAMILIAS: dict[str, type[Estrategia]] = {}


def registrar(clase: type[Estrategia]) -> type[Estrategia]:
    """Decorador que añade una familia de estrategias al registro."""
    FAMILIAS[clase.familia] = clase
    return clase


def crear(familia: str, parametros: dict[str, Any]) -> Estrategia:
    """Instancia una estrategia a partir de su familia y parámetros."""
    if familia not in FAMILIAS:
        raise KeyError(f"Familia desconocida: {familia}. Disponibles: {sorted(FAMILIAS)}")
    return FAMILIAS[familia](**parametros)


def hash_estrategia(familia: str, parametros: dict[str, Any]) -> str:
    """Identificador estable de una combinación familia + parámetros."""
    texto = familia + "|" + "|".join(f"{k}={parametros[k]}" for k in sorted(parametros))
    return hashlib.sha1(texto.encode()).hexdigest()[:12]


def expandir(especificacion: Any) -> list:
    """Convierte la especificación de un parámetro de config.yaml en su lista de valores.

    - `[mín, máx, paso]` numérico → rango (enteros si todo son enteros).
    - Lista de textos → categorías.
    """
    if (
        isinstance(especificacion, list)
        and len(especificacion) == 3
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in especificacion)
    ):
        minimo, maximo, paso = especificacion
        if all(isinstance(v, int) for v in especificacion):
            return list(range(int(minimo), int(maximo) + 1, int(paso)))
        valores = np.arange(minimo, maximo + paso / 2, paso)
        return [round(float(v), 6) for v in valores]
    if isinstance(especificacion, list):
        return list(especificacion)
    return [especificacion]


class Estrategia(ABC):
    """Estrategia = familia + parámetros → posiciones objetivo."""

    familia: ClassVar[str] = ""
    nombre_legible: ClassVar[str] = ""
    descripcion: ClassVar[str] = ""
    admite_cortos: ClassVar[bool] = False

    def __init__(self, **parametros: Any) -> None:
        self.parametros = parametros

    # ------------------------------------------------------------------ espacio
    @classmethod
    def valida(cls, parametros: dict[str, Any]) -> bool:
        """Filtra combinaciones sin sentido (p. ej. media rápida >= media lenta)."""
        return True

    @classmethod
    def rejilla(cls, espacio: dict[str, Any]) -> list[dict[str, Any]]:
        """Todas las combinaciones válidas del espacio de parámetros."""
        claves = list(espacio)
        valores = [expandir(espacio[c]) for c in claves]
        combos = (dict(zip(claves, v)) for v in itertools.product(*valores))
        return [c for c in combos if cls.valida(c)]

    # ------------------------------------------------------------------ señales
    @abstractmethod
    def _posiciones(self, c: CacheIndicadores, cortos: bool) -> np.ndarray:
        """Implementación concreta: devuelve posiciones (float) de la misma longitud que los datos."""

    def posiciones(self, datos: pd.DataFrame | CacheIndicadores, cortos: bool = False) -> np.ndarray:
        """Posiciones objetivo decididas al cierre de cada vela (NaN → 0)."""
        c = datos if isinstance(datos, CacheIndicadores) else CacheIndicadores(datos)
        pos = np.asarray(self._posiciones(c, cortos and self.admite_cortos), dtype=np.float64)
        return np.nan_to_num(pos, nan=0.0)

    # ------------------------------------------------------------------ texto
    def descripcion_corta(self) -> str:
        """Nombre breve legible, p. ej. 'Cruce de medias (rapida=10, lenta=50)'."""
        texto = ", ".join(f"{k}={v}" for k, v in self.parametros.items())
        return f"{self.nombre_legible} ({texto})"

    def hash(self) -> str:
        return hash_estrategia(self.familia, self.parametros)

    def __repr__(self) -> str:
        return f"<{self.familia} {self.parametros}>"
