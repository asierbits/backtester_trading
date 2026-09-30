"""Cruce de medias: largo mientras la media rápida está por encima de la lenta."""

from __future__ import annotations

from typing import Any

import numpy as np

from modulos.trading.estrategias.base import Estrategia, registrar
from modulos.trading.estrategias.indicadores import CacheIndicadores


@registrar
class CruceMedias(Estrategia):
    familia = "cruce_medias"
    nombre_legible = "Cruce de medias"
    descripcion = (
        "Compra cuando la media rápida cruza por encima de la lenta y vende cuando cruza por "
        "debajo. Con cortos, se pone corta mientras la rápida esté por debajo."
    )
    admite_cortos = True

    @classmethod
    def valida(cls, p: dict[str, Any]) -> bool:
        return p["rapida"] < p["lenta"]

    def _posiciones(self, c: CacheIndicadores, cortos: bool) -> np.ndarray:
        p = self.parametros
        rapida = c.media(p["tipo_media"], p["rapida"])
        lenta = c.media(p["tipo_media"], p["lenta"])
        largo = (rapida > lenta).astype(np.float64)
        if cortos:
            return largo - (rapida < lenta).astype(np.float64)
        return largo
