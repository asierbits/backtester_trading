"""Motor de backtest vectorizado/compilado con numba.

Reglas (honestas por diseño):
1. La posición objetivo decidida al cierre de la vela t se ejecuta en la vela t+1
   (en su apertura por defecto, o en su cierre). Nunca en la misma vela.
2. Cada ejecución paga comisión y slippage.
3. Sin apalancamiento: como mucho el 100 % del capital, largo o corto.
4. El tamaño se fija al entrar (100 % del capital o sizing por volatilidad) y no se
   reequilibra dentro de la operación (evita costes irreales).
5. La curva de capital se valora a precio de cierre de cada vela (mark-to-market),
   incluida la operación abierta.
6. Se registra cada operación: entrada, salida, precios, comisión, resultado y duración.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import njit

from modulos.trading.backtest.costes import Costes
from modulos.trading.backtest.metricas import (
    CLAVES_METRICAS,
    rentabilidades_por_vela,
    resumen_numba,
)
from modulos.trading.datos.limpieza import VELAS_POR_ANO


@njit(cache=True)
def simular(
    ejec: np.ndarray,
    close: np.ndarray,
    objetivo: np.ndarray,
    ini: int,
    fin: int,
    capital: float,
    comision: float,
    slip: np.ndarray,
):
    """Núcleo del backtest. Devuelve equity y arrays de operaciones.

    `objetivo[i]` es la posición deseada al cierre de i (signo = dirección, módulo =
    fracción del capital al entrar). Se ejecuta en la vela i+1 a `ejec[i+1]`.
    Si ini > 0, la primera vela del tramo ya puede ejecutar la señal de ini-1
    (la señal se calculó con datos anteriores, así que no hay sesgo).
    """
    n = fin - ini
    equity = np.empty(n)
    t_ent = np.empty(n + 1, dtype=np.int64)
    t_sal = np.empty(n + 1, dtype=np.int64)
    t_pent = np.empty(n + 1)
    t_psal = np.empty(n + 1)
    t_dir = np.empty(n + 1, dtype=np.int64)
    t_com = np.empty(n + 1)
    t_pnl = np.empty(n + 1)
    t_pct = np.empty(n + 1)

    cash = capital
    unidades = 0.0
    direccion = 0
    n_tr = 0
    expuestas = 0
    ent_idx = 0
    ent_px = 0.0
    ent_com = 0.0
    ent_cash = 0.0
    arruinado = False

    for i in range(ini, fin):
        k = i - ini
        if arruinado:
            equity[k] = 0.0
            continue
        deseado = objetivo[i - 1] if i >= 1 else 0.0
        signo = 0
        if deseado > 0:
            signo = 1
        elif deseado < 0:
            signo = -1

        if signo != direccion:
            # 1) Cerrar la posición actual
            if direccion != 0:
                px = ejec[i] * (1.0 - slip[i] * direccion)
                com = abs(unidades) * px * comision
                cash += unidades * px - com
                t_ent[n_tr] = ent_idx
                t_sal[n_tr] = i
                t_pent[n_tr] = ent_px
                t_psal[n_tr] = px
                t_dir[n_tr] = direccion
                t_com[n_tr] = ent_com + com
                t_pnl[n_tr] = cash - ent_cash
                t_pct[n_tr] = (cash - ent_cash) / ent_cash if ent_cash > 0 else 0.0
                n_tr += 1
                unidades = 0.0
                direccion = 0
            # 2) Abrir la nueva posición
            if signo != 0 and cash > 0:
                px = ejec[i] * (1.0 + slip[i] * signo)
                fraccion = min(abs(deseado), 1.0)
                nocional = cash * fraccion / (1.0 + comision)
                com = nocional * comision
                ent_cash = cash
                cash -= signo * nocional + com
                unidades = signo * nocional / px
                direccion = signo
                ent_idx = i
                ent_px = px
                ent_com = com

        valor = cash + unidades * close[i]
        if valor <= 0.0:
            # Ruina (solo posible con cortos): se liquida todo
            arruinado = True
            valor = 0.0
        equity[k] = valor
        if direccion != 0:
            expuestas += 1

    # Operación abierta al final: se valora a mercado (sin comisión de salida)
    if direccion != 0 and not arruinado:
        t_ent[n_tr] = ent_idx
        t_sal[n_tr] = -1
        t_pent[n_tr] = ent_px
        t_psal[n_tr] = close[fin - 1]
        t_dir[n_tr] = direccion
        t_com[n_tr] = ent_com
        t_pnl[n_tr] = equity[n - 1] - ent_cash
        t_pct[n_tr] = (equity[n - 1] - ent_cash) / ent_cash if ent_cash > 0 else 0.0
        n_tr += 1

    return equity, t_ent, t_sal, t_pent, t_psal, t_dir, t_com, t_pnl, t_pct, n_tr, expuestas


def estimar_velas_ano(indice: pd.DatetimeIndex, temporalidad: str | None = None) -> float:
    """Velas por año reales de la serie (24/7 en cripto, ~252 días en acciones)."""
    if len(indice) > 60:
        anos = (indice[-1] - indice[0]).total_seconds() / (365.25 * 86400)
        if anos > 0.25:
            return float(len(indice) / anos)
    return float(VELAS_POR_ANO.get(temporalidad or "1d", 365))


@dataclass
class Mercado:
    """Arrays de un activo listos para el motor."""

    df: pd.DataFrame
    temporalidad: str
    precio_ejecucion: str = "apertura"
    velas_ano: float = field(init=False)
    ejec: np.ndarray = field(init=False)
    close: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.close = self.df["close"].to_numpy(dtype=np.float64)
        columna = "open" if self.precio_ejecucion == "apertura" else "close"
        self.ejec = self.df[columna].to_numpy(dtype=np.float64)
        self.velas_ano = estimar_velas_ano(self.df.index, self.temporalidad)

    @property
    def n(self) -> int:
        return len(self.close)

    @property
    def indice(self) -> pd.DatetimeIndex:
        return self.df.index


@dataclass
class Resultado:
    """Resultado completo de un backtest."""

    equity: pd.Series
    operaciones: pd.DataFrame
    metricas: dict[str, float]
    capital: float

    @property
    def rentabilidades(self) -> pd.Series:
        """Rentabilidad de cada vela (la primera, respecto al capital inicial)."""
        r = rentabilidades_por_vela(self.equity.to_numpy(), self.capital)
        return pd.Series(r, index=self.equity.index)


def backtest_rapido(
    mercado: Mercado,
    objetivo: np.ndarray,
    slip: np.ndarray,
    comision: float,
    capital: float,
    ini: int,
    fin: int,
) -> np.ndarray:
    """Solo métricas (array en el orden de CLAVES_METRICAS). Para probar miles de estrategias."""
    eq, _, _, _, _, _, _, _, pct, n_tr, expo = simular(
        mercado.ejec, mercado.close, objetivo, ini, fin, capital, comision, slip
    )
    return resumen_numba(eq, capital, pct, n_tr, expo, mercado.velas_ano)


def backtest(
    mercado: Mercado,
    objetivo: np.ndarray,
    costes: Costes,
    capital: float,
    ini: int = 0,
    fin: int | None = None,
    slip: np.ndarray | None = None,
) -> Resultado:
    """Backtest completo con curva de capital y lista de operaciones."""
    fin = mercado.n if fin is None else fin
    if slip is None:
        slip = costes.array_slippage(mercado.df)
    eq, ent, sal, pent, psal, dirc, com, pnl, pct, n_tr, expo = simular(
        mercado.ejec, mercado.close, objetivo, ini, fin, capital, costes.comision, slip
    )
    indice = mercado.indice
    equity = pd.Series(eq, index=indice[ini:fin], name="equity")
    salidas = sal[:n_tr]
    operaciones = pd.DataFrame(
        {
            "entrada": indice[ent[:n_tr]],
            "salida": [indice[s] if s >= 0 else pd.NaT for s in salidas],
            "direccion": dirc[:n_tr],
            "precio_entrada": pent[:n_tr],
            "precio_salida": psal[:n_tr],
            "comision": com[:n_tr],
            "resultado": pnl[:n_tr],
            "resultado_pct": pct[:n_tr],
            "duracion_velas": np.where(salidas >= 0, salidas - ent[:n_tr], fin - 1 - ent[:n_tr]),
            "abierta": salidas < 0,
        }
    )
    valores = resumen_numba(eq, capital, pct, n_tr, expo, mercado.velas_ano)
    metricas = metricas_a_dict(valores)
    return Resultado(equity=equity, operaciones=operaciones, metricas=metricas, capital=capital)


def metricas_a_dict(valores: np.ndarray) -> dict[str, float]:
    """Convierte el array de métricas en diccionario."""
    return {k: float(v) for k, v in zip(CLAVES_METRICAS, valores)}


def aplicar_sizing_volatilidad(
    objetivo: np.ndarray, close: np.ndarray, velas_ano: float, vol_objetivo: float
) -> np.ndarray:
    """Escala la posición para apuntar a una volatilidad anual (máximo 100 % del capital).

    Usa la volatilidad realizada hasta el cierre de t (EWM de 30 velas): causal.
    """
    rent = pd.Series(close).pct_change()
    vol = rent.ewm(span=30, min_periods=30).std().to_numpy() * np.sqrt(velas_ano)
    fraccion = np.clip(vol_objetivo / np.where(vol > 0, vol, np.nan), 0.0, 1.0)
    fraccion = np.nan_to_num(fraccion, nan=1.0)
    return objetivo * fraccion
