"""Estrategias combinadas: filtro de tendencia (precio sobre SMA larga) + señal de otra familia."""

from __future__ import annotations

from typing import Any

import numpy as np

from modulos.trading.estrategias.base import FAMILIAS, Estrategia, expandir, registrar
from modulos.trading.estrategias.indicadores import CacheIndicadores


@registrar
class Combinada(Estrategia):
    familia = "combinada"
    nombre_legible = "Combinada (filtro de tendencia)"
    descripcion = (
        "Usa la señal de otra familia, pero solo permite estar largo si el precio está por "
        "encima de la SMA de tendencia (y corto solo si está por debajo)."
    )
    admite_cortos = True

    @classmethod
    def rejilla_con_bases(
        cls, espacio: dict[str, Any], espacios_base: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Combina cada base con cada periodo de filtro. Los parámetros de la base van con prefijo."""
        combos: list[dict[str, Any]] = []
        filtros = expandir(espacio["filtro"])
        for base in espacio["bases"]:
            if base not in FAMILIAS or base == cls.familia or base not in espacios_base:
                continue
            for p_base in FAMILIAS[base].rejilla(espacios_base[base]):
                for f in filtros:
                    combo = {"base": base, "filtro": f}
                    combo.update({f"b_{k}": v for k, v in p_base.items()})
                    combos.append(combo)
        return combos

    def parametros_base(self) -> dict[str, Any]:
        return {k[2:]: v for k, v in self.parametros.items() if k.startswith("b_")}

    def _posiciones(self, c: CacheIndicadores, cortos: bool) -> np.ndarray:
        base = FAMILIAS[self.parametros["base"]](**self.parametros_base())
        pos = base.posiciones(c, cortos)
        tendencia = c.media("sma", self.parametros["filtro"])
        alcista = c.close > tendencia
        bajista = c.close < tendencia
        return np.where(pos > 0, pos * alcista, np.where(pos < 0, pos * bajista, 0.0))

    def descripcion_corta(self) -> str:
        base = FAMILIAS[self.parametros["base"]](**self.parametros_base())
        return f"{base.descripcion_corta()} + tendencia SMA {self.parametros['filtro']}"
