"""Cálculos bajo demanda para la página de detalle y los informes.

Las curvas de capital no se guardan en la base de datos (ocuparían mucho): se
recalculan con los mismos datos, parámetros y motor, así que salen idénticas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from modulos.trading.backtest.motor import Resultado, backtest
from modulos.trading.estrategias import FAMILIAS, crear
from modulos.trading.estrategias.base import expandir
from modulos.trading.experimento import (
    ContextoActivo,
    _espacio_familia,
    calcular_posiciones,
    contexto_de_estrategia,
    metricas_tramo,
)
from modulos.trading.validacion import mapa_calor as mc


@dataclass
class Detalle:
    info: dict
    cfg: dict
    ctx: ContextoActivo
    resultado: Resultado
    buy_hold: Resultado
    fecha_corte: pd.Timestamp
    nombre: str


def detalle(estrategia_id: int) -> Detalle:
    info, cfg, ctx, _ = contexto_de_estrategia(estrategia_id)
    pos = calcular_posiciones(info["familia"], info["parametros"], ctx, cfg)
    res = backtest(ctx.mercado, pos, ctx.costes, ctx.capital, 0, ctx.n, ctx.slip)
    bh = backtest(ctx.mercado, np.ones(ctx.n), ctx.costes, ctx.capital, 0, ctx.n, ctx.slip)
    nombre = crear(info["familia"], info["parametros"]).descripcion_corta()
    return Detalle(info, cfg, ctx, res, bh, ctx.mercado.indice[ctx.corte], nombre)


def parametros_numericos(d: Detalle) -> list[str]:
    """Parámetros que se pueden poner en los ejes del mapa de calor."""
    espacio = _espacio_familia(d.info["familia"], d.info["parametros"], d.cfg)
    salida = []
    for clave, spec in espacio.items():
        valores = expandir(spec)
        if len(valores) > 1 and all(isinstance(v, (int, float)) for v in valores):
            salida.append(clave)
    return salida


def mapa_calor(d: Detalle, param_x: str, param_y: str, maximo_por_eje: int = 15) -> pd.DataFrame:
    """Sharpe en entrenamiento variando dos parámetros (el resto fijos)."""
    familia = d.info["familia"]
    espacio = _espacio_familia(familia, d.info["parametros"], d.cfg)

    def evaluar(p: dict[str, Any]) -> float | None:
        if familia == "combinada":
            base = {k[2:]: v for k, v in p.items() if k.startswith("b_")}
            if not FAMILIAS[p["base"]].valida(base):
                return None
        elif not FAMILIAS[familia].valida(p):
            return None
        pos = calcular_posiciones(familia, p, d.ctx, d.cfg)
        return float(metricas_tramo(d.ctx, pos, "train")[3])

    return mc.mapa_calor(d.info["parametros"], espacio, param_x, param_y, evaluar, maximo_por_eje)
