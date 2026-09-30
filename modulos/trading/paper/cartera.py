"""Optimización de cartera entre estrategias (Fase 3).

Combina varias estrategias (normalmente las APROBADAS) repartiendo el capital con:
- **igual**: mismo peso para todas.
- **inverso_volatilidad**: más peso a las menos volátiles.
- **minima_varianza**: pesos que minimizan la varianza de la cartera (solo largos,
  con la matriz de covarianzas encogida para que sea estable).

Los pesos se calculan con el tramo de ENTRENAMIENTO y se evalúan en el de TEST.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modulos.trading.backtest.motor import backtest
from modulos.trading.experimento import calcular_posiciones, contexto_de_estrategia

METODOS = {
    "igual": "Pesos iguales",
    "inverso_volatilidad": "Inverso de la volatilidad",
    "minima_varianza": "Mínima varianza",
}


def rentabilidades_diarias(estrategia_id: int) -> tuple[pd.Series, pd.Timestamp, str]:
    """Rentabilidad diaria de la estrategia en toda la historia, y fecha de inicio del test."""
    info, cfg, ctx, _ = contexto_de_estrategia(estrategia_id)
    pos = calcular_posiciones(info["familia"], info["parametros"], ctx, cfg)
    res = backtest(ctx.mercado, pos, ctx.costes, ctx.capital, 0, ctx.n, ctx.slip)
    diaria = res.equity.resample("1D").last().dropna().pct_change().dropna()
    nombre = f"#{estrategia_id} {info['familia']} {info['activo']}"
    return diaria, ctx.mercado.indice[ctx.corte], nombre


def pesos(rent: pd.DataFrame, metodo: str) -> pd.Series:
    """Pesos (suman 1) a partir de rentabilidades diarias (columnas = estrategias)."""
    n = rent.shape[1]
    if n == 1 or metodo == "igual":
        return pd.Series(1 / n, index=rent.columns)
    vol = rent.std().replace(0, np.nan)
    if metodo == "inverso_volatilidad":
        w = (1 / vol).fillna(0)
        return w / w.sum()
    if metodo == "minima_varianza":
        cov = rent.cov().to_numpy()
        # Encogimiento hacia la diagonal (Ledoit-Wolf simplificado, 20 %)
        cov = 0.8 * cov + 0.2 * np.diag(np.diag(cov))
        try:
            inv = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            return pd.Series(1 / n, index=rent.columns)
        w = np.clip(inv @ np.ones(n), 0, None)
        if w.sum() <= 0:
            return pd.Series(1 / n, index=rent.columns)
        return pd.Series(w / w.sum(), index=rent.columns)
    raise ValueError(f"Método desconocido: {metodo}")


def construir_cartera(ids: list[int], metodo: str = "inverso_volatilidad") -> dict:
    """Pesos con train, curva y métricas en test (rebalanceo diario implícito)."""
    series, cortes, nombres = [], [], []
    for i in ids:
        r, corte, nombre = rentabilidades_diarias(i)
        series.append(r.rename(nombre))
        cortes.append(corte)
        nombres.append(nombre)
    rent = pd.concat(series, axis=1).dropna()
    corte = max(cortes)
    train, test = rent[rent.index < corte], rent[rent.index >= corte]
    if len(train) < 30 or len(test) < 30:
        raise ValueError("No hay suficientes días en común entre las estrategias elegidas")
    w = pesos(train, metodo)
    r_test = test @ w
    curva = (1 + r_test).cumprod()
    anual = 365 if (test.index.dayofweek >= 5).any() else 252
    vol = r_test.std() * np.sqrt(anual)
    pico = curva.cummax()
    return {
        "pesos": w,
        "curva": curva,
        "curvas_individuales": (1 + test).cumprod(),
        "correlaciones": train.corr(),
        "metricas": {
            "rentabilidad": float(curva.iloc[-1] - 1),
            "volatilidad": float(vol),
            "sharpe": float(r_test.mean() / r_test.std() * np.sqrt(anual)) if r_test.std() > 0 else 0.0,
            "max_dd": float((1 - curva / pico).max()),
        },
    }
