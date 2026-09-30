"""Indicadores técnicos implementados a mano con pandas/numpy.

Todos son **causales**: el valor en la vela t solo usa datos hasta el cierre de t
(nunca datos futuros). Los tests de `tests/test_sin_sesgo_futuro.py` lo comprueban
modificando el futuro y verificando que el pasado no cambia.

`CacheIndicadores` guarda cada indicador ya calculado para un activo, de modo que
miles de estrategias que comparten la misma media de 20 velas la calculan una vez.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


def sma(x: pd.Series, n: int) -> pd.Series:
    """Media móvil simple de n velas."""
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    """Media móvil exponencial (span n), sin ajuste: recursiva y causal."""
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    """RSI de Wilder (suavizado exponencial con alfa = 1/n)."""
    delta = close.diff()
    ganancia = delta.clip(lower=0.0)
    perdida = (-delta).clip(lower=0.0)
    media_g = ganancia.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    media_p = perdida.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = media_g / media_p.replace(0.0, np.nan)
    salida = 100 - 100 / (1 + rs)
    # Sin pérdidas en la ventana → RSI = 100
    salida = salida.where(~((media_p == 0) & media_g.notna()), 100.0)
    return salida


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range de Wilder."""
    cierre_prev = df["close"].shift(1)
    rango = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - cierre_prev).abs(),
            (df["low"] - cierre_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return rango.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.Series, n: int, k: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bandas de Bollinger: (media, banda superior, banda inferior)."""
    media = sma(close, n)
    desv = close.rolling(n, min_periods=n).std(ddof=0)
    return media, media + k * desv, media - k * desv


def macd(close: pd.Series, rapida: int, lenta: int, senal: int) -> tuple[pd.Series, pd.Series]:
    """MACD clásico: (línea MACD, línea de señal)."""
    linea = ema(close, rapida) - ema(close, lenta)
    return linea, linea.ewm(span=senal, adjust=False, min_periods=senal).mean()


def donchian_alto(high: pd.Series, n: int) -> pd.Series:
    """Máximo de las n velas ANTERIORES (excluye la vela actual para poder hablar de 'ruptura')."""
    return high.rolling(n, min_periods=n).max().shift(1)


def donchian_bajo(low: pd.Series, n: int) -> pd.Series:
    """Mínimo de las n velas ANTERIORES."""
    return low.rolling(n, min_periods=n).min().shift(1)


# --------------------------------------------------------------------------- numba
@njit(cache=True)
def maquina_estados(entrada: np.ndarray, salida: np.ndarray) -> np.ndarray:
    """Posición 0/1 a partir de condiciones de entrada y salida (solo largos).

    Fuera de mercado → entra cuando `entrada` es cierta. Dentro → sale cuando
    `salida` es cierta. La decisión en t solo usa las condiciones hasta t.
    """
    n = entrada.shape[0]
    pos = np.zeros(n, dtype=np.float64)
    actual = 0.0
    for i in range(n):
        if actual == 0.0:
            if entrada[i]:
                actual = 1.0
        elif salida[i]:
            actual = 0.0
        pos[i] = actual
    return pos


@njit(cache=True)
def maquina_estados_lc(
    ent_largo: np.ndarray, sal_largo: np.ndarray, ent_corto: np.ndarray, sal_corto: np.ndarray
) -> np.ndarray:
    """Igual que `maquina_estados` pero con cortos (-1). Una señal contraria da la vuelta."""
    n = ent_largo.shape[0]
    pos = np.zeros(n, dtype=np.float64)
    actual = 0.0
    for i in range(n):
        if actual == 1.0:
            if ent_corto[i]:
                actual = -1.0
            elif sal_largo[i]:
                actual = 0.0
        elif actual == -1.0:
            if ent_largo[i]:
                actual = 1.0
            elif sal_corto[i]:
                actual = 0.0
        else:
            if ent_largo[i]:
                actual = 1.0
            elif ent_corto[i]:
                actual = -1.0
        pos[i] = actual
    return pos


class CacheIndicadores:
    """Calcula y memoriza indicadores de un DataFrame OHLCV (arrays numpy)."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.close_s = df["close"]
        self.close = df["close"].to_numpy(dtype=np.float64)
        self._cache: dict[tuple, object] = {}

    def _memo(self, clave: tuple, funcion):  # noqa: ANN001, ANN202
        if clave not in self._cache:
            self._cache[clave] = funcion()
        return self._cache[clave]

    def media(self, tipo: str, n: int) -> np.ndarray:
        """SMA o EMA de n velas."""
        f = sma if tipo == "sma" else ema
        return self._memo((tipo, n), lambda: f(self.close_s, int(n)).to_numpy())

    def rsi(self, n: int) -> np.ndarray:
        return self._memo(("rsi", n), lambda: rsi(self.close_s, int(n)).to_numpy())

    def donchian_alto(self, n: int) -> np.ndarray:
        return self._memo(("dalto", n), lambda: donchian_alto(self.df["high"], int(n)).to_numpy())

    def donchian_bajo(self, n: int) -> np.ndarray:
        return self._memo(("dbajo", n), lambda: donchian_bajo(self.df["low"], int(n)).to_numpy())

    def bollinger(self, n: int, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def calc() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            media = self.media("sma", n)
            desv = self._memo(
                ("std", n), lambda: self.close_s.rolling(int(n), min_periods=int(n)).std(ddof=0).to_numpy()
            )
            return media, media + k * desv, media - k * desv

        return self._memo(("boll", n, round(float(k), 4)), calc)

    def macd(self, rapida: int, lenta: int, senal: int) -> tuple[np.ndarray, np.ndarray]:
        def calc() -> tuple[np.ndarray, np.ndarray]:
            linea = self.media("ema", rapida) - self.media("ema", lenta)
            s = pd.Series(linea).ewm(span=int(senal), adjust=False, min_periods=int(senal)).mean()
            return linea, s.to_numpy()

        return self._memo(("macd", rapida, lenta, senal), calc)

    def atr_pct(self, n: int = 14) -> np.ndarray:
        """ATR relativo al precio (fracción)."""
        return self._memo(("atr_pct", n), lambda: (atr(self.df, n) / self.close_s).to_numpy())
