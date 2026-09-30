"""Ruptura de canal Donchian (estilo 'tortugas')."""

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
class RupturaDonchian(Estrategia):
    familia = "ruptura_donchian"
    nombre_legible = "Ruptura Donchian"
    descripcion = (
        "Compra cuando el cierre supera el máximo de las N velas anteriores y vende cuando "
        "cae por debajo del mínimo de las M velas anteriores."
    )
    admite_cortos = True

    @classmethod
    def valida(cls, p: dict[str, Any]) -> bool:
        return p["salida"] < p["entrada"]

    def _posiciones(self, c: CacheIndicadores, cortos: bool) -> np.ndarray:
        p = self.parametros
        close = c.close
        entrada = close > c.donchian_alto(p["entrada"])
        salida = close < c.donchian_bajo(p["salida"])
        if cortos:
            ent_corto = close < c.donchian_bajo(p["entrada"])
            sal_corto = close > c.donchian_alto(p["salida"])
            return maquina_estados_lc(entrada, salida, ent_corto, sal_corto)
        return maquina_estados(entrada, salida)
