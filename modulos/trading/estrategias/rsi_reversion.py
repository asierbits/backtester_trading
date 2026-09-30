"""RSI de reversión a la media: compra en sobreventa, vende en sobrecompra."""

from __future__ import annotations

from typing import Any

import numpy as np

from modulos.trading.estrategias.base import Estrategia, registrar
from modulos.trading.estrategias.indicadores import (
    CacheIndicadores,
    maquina_estados,
    maquina_estados_lc,
)


@registrar
class RsiReversion(Estrategia):
    familia = "rsi_reversion"
    nombre_legible = "RSI reversión"
    descripcion = (
        "Entra largo cuando el RSI cae por debajo del umbral de compra (sobreventa) y sale "
        "cuando supera el umbral de venta (sobrecompra)."
    )
    admite_cortos = True

    @classmethod
    def valida(cls, p: dict[str, Any]) -> bool:
        return p["compra"] < p["venta"]

    def _posiciones(self, c: CacheIndicadores, cortos: bool) -> np.ndarray:
        p = self.parametros
        r = c.rsi(p["periodo"])
        compra = r < p["compra"]
        venta = r > p["venta"]
        if cortos:
            return maquina_estados_lc(compra, venta, venta, compra)
        return maquina_estados(compra, venta)
