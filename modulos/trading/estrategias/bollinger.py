"""Bandas de Bollinger: reversión a la media o ruptura."""

from __future__ import annotations

import numpy as np

from modulos.trading.estrategias.base import Estrategia, registrar
from modulos.trading.estrategias.indicadores import (
    CacheIndicadores,
    maquina_estados,
    maquina_estados_lc,
)


@registrar
class Bollinger(Estrategia):
    familia = "bollinger"
    nombre_legible = "Bandas de Bollinger"
    descripcion = (
        "Modo 'reversion': compra al cerrar por debajo de la banda inferior y sale al volver a "
        "la media. Modo 'ruptura': compra al cerrar por encima de la banda superior y sale al "
        "perder la media."
    )
    admite_cortos = True

    def _posiciones(self, c: CacheIndicadores, cortos: bool) -> np.ndarray:
        p = self.parametros
        media, sup, inf = c.bollinger(p["periodo"], p["desviaciones"])
        close = c.close
        if p["modo"] == "reversion":
            ent_l, sal_l = close < inf, close > media
            ent_c, sal_c = close > sup, close < media
        else:
            ent_l, sal_l = close > sup, close < media
            ent_c, sal_c = close < inf, close > media
        if cortos:
            return maquina_estados_lc(ent_l, sal_l, ent_c, sal_c)
        return maquina_estados(ent_l, sal_l)
