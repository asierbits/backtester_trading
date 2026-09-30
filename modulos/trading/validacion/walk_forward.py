"""Walk-forward: optimizar en una ventana, validar en la siguiente, avanzar y repetir.

Se divide la historia en (k+1) tramos consecutivos. En cada paso se re-optimiza la
familia de la estrategia con el tramo anterior (eligiendo los parámetros con mejor
Sharpe entre un conjunto de candidatos) y se evalúa en el tramo siguiente, que no se
ha visto. La curva de capital encadenada solo usa tramos fuera de muestra.

También se evalúan los parámetros fijos del finalista en cada tramo, para ver si su
resultado viene de una sola racha o es consistente en el tiempo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from modulos.trading.backtest.metricas import CLAVES_METRICAS, rentabilidades_por_vela
from modulos.trading.backtest.motor import Mercado, backtest_rapido, simular

I_SHARPE = CLAVES_METRICAS.index("sharpe")
I_RENT = CLAVES_METRICAS.index("rentabilidad")
I_NOPS = CLAVES_METRICAS.index("n_ops")


def limites_ventanas(n: int, k: int) -> list[tuple[int, int, int]]:
    """Lista de (inicio_opt, inicio_test, fin_test) para k ventanas fuera de muestra."""
    bordes = np.linspace(0, n, k + 2).astype(int)
    return [(int(bordes[i]), int(bordes[i + 1]), int(bordes[i + 2])) for i in range(k)]


def walk_forward(
    mercado: Mercado,
    posiciones_candidatos: list[np.ndarray],
    parametros_candidatos: list[dict[str, Any]],
    indice_finalista: int,
    comision: float,
    slip: np.ndarray,
    capital: float,
    n_ventanas: int,
    min_ops_por_ventana: int = 3,
    min_rentables: float = 0.5,
    posiciones_bh: np.ndarray | None = None,
) -> dict:
    """Ejecuta el walk-forward y devuelve un resumen serializable."""
    ventanas = []
    tramos_oos: list[np.ndarray] = []
    fechas_oos: list[pd.DatetimeIndex] = []
    for ini_opt, ini_test, fin_test in limites_ventanas(mercado.n, n_ventanas):
        # 1) Optimización en la ventana anterior
        mejor, mejor_sr = indice_finalista, -np.inf
        for j, pos in enumerate(posiciones_candidatos):
            m = backtest_rapido(mercado, pos, slip, comision, capital, ini_opt, ini_test)
            if m[I_NOPS] >= min_ops_por_ventana and m[I_SHARPE] > mejor_sr:
                mejor, mejor_sr = j, m[I_SHARPE]
        # 2) Validación en la ventana siguiente (fuera de muestra)
        eq = simular(
            mercado.ejec,
            mercado.close,
            posiciones_candidatos[mejor],
            ini_test,
            fin_test,
            capital,
            comision,
            slip,
        )[0]
        fijo = backtest_rapido(
            mercado, posiciones_candidatos[indice_finalista], slip, comision, capital, ini_test, fin_test
        )
        bh = (
            backtest_rapido(mercado, posiciones_bh, slip, comision, capital, ini_test, fin_test)
            if posiciones_bh is not None
            else None
        )
        oos = eq[-1] / capital - 1
        tramos_oos.append(rentabilidades_por_vela(eq, capital))
        fechas_oos.append(mercado.indice[ini_test:fin_test])
        ventanas.append(
            {
                "desde": str(mercado.indice[ini_test].date()),
                "hasta": str(mercado.indice[fin_test - 1].date()),
                "parametros_elegidos": parametros_candidatos[mejor],
                "sharpe_optimizacion": float(mejor_sr) if np.isfinite(mejor_sr) else None,
                "rentabilidad_oos": float(oos),
                "rentabilidad_fijos": float(fijo[I_RENT]),
                "sharpe_fijos": float(fijo[I_SHARPE]),
                "rentabilidad_bh": float(bh[I_RENT]) if bh is not None else None,
            }
        )
    r = np.concatenate(tramos_oos)
    curva = capital * np.cumprod(1 + r)
    desv = r.std()
    sharpe_oos = float(r.mean() / desv * np.sqrt(mercado.velas_ano)) if desv > 0 else 0.0
    pct_rentables = float(np.mean([v["rentabilidad_oos"] > 0 for v in ventanas]))
    pct_fijos = float(np.mean([v["rentabilidad_fijos"] > 0 for v in ventanas]))
    fechas = pd.DatetimeIndex(np.concatenate([f.to_numpy() for f in fechas_oos]))
    # Curva reducida para gráficas (máx. 400 puntos)
    paso = max(1, len(curva) // 400)
    return {
        "ventanas": ventanas,
        "pct_ventanas_rentables": pct_rentables,
        "pct_ventanas_rentables_fijos": pct_fijos,
        "rentabilidad_oos_total": float(curva[-1] / capital - 1),
        "sharpe_oos": sharpe_oos,
        "curva": {
            "fechas": [str(f) for f in fechas[::paso]],
            "equity": [float(x) for x in curva[::paso]],
        },
        "supera": bool(pct_rentables >= min_rentables and curva[-1] > capital),
    }


def candidatos_familia(
    combinaciones: list[dict[str, Any]],
    parametros_finalista: dict[str, Any],
    maximo: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Subconjunto de combinaciones de la familia; el finalista siempre va el primero."""
    otros = [c for c in combinaciones if c != parametros_finalista]
    if len(otros) > maximo - 1:
        idx = np.sort(rng.choice(len(otros), size=maximo - 1, replace=False))
        otros = [otros[i] for i in idx]
    return [parametros_finalista, *otros]
