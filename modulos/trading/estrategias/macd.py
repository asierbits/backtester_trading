"""MACD: largo mientras la línea MACD está por encima de su señal."""

from __future__ import annotations

from typing import Any

import numpy as np

from modulos.trading.estrategias.base import Estrategia, registrar
from modulos.trading.estrategias.indicadores import CacheIndicadores


@registrar
class Macd(Estrategia):
    familia = "macd"
    nombre_legible = "MACD"
    descripcion = "Largo mientras la línea MACD (EMA rápida − EMA lenta) está por encima de su señal."
    admite_cortos = True

    @classmethod
    def valida(cls, p: dict[str, Any]) -> bool:
        return p["rapida"] < p["lenta"]

    def _posiciones(self, c: CacheIndicadores, cortos: bool) -> np.ndarray:
        p = self.parametros
        linea, senal = c.macd(p["rapida"], p["lenta"], p["senal"])
        largo = (linea > senal).astype(np.float64)
        if cortos:
            return largo - (linea < senal).astype(np.float64)
        return largo
