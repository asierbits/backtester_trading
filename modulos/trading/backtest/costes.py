"""Comisiones y deslizamiento (slippage).

- Comisión: fracción del importe de cada ejecución (entrada y salida), por defecto 0,1 %.
- Slippage: el precio de ejecución empeora una fracción (compras más caro, vendes más barato).
  Puede ser fijo o depender de la volatilidad (ATR) o del volumen de la vela anterior,
  siempre con información conocida ANTES de ejecutar (sin sesgo de futuro).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from modulos.trading.estrategias.indicadores import atr


@dataclass(frozen=True)
class Costes:
    """Parámetros de costes de ejecución."""

    comision: float = 0.001
    slippage: float = 0.0005
    modo_slippage: str = "fijo"  # fijo | volatilidad | volumen

    @classmethod
    def desde_config(cls, cfg_backtest: dict[str, Any]) -> Costes:
        return cls(
            comision=float(cfg_backtest.get("comision", 0.001)),
            slippage=float(cfg_backtest.get("slippage", 0.0005)),
            modo_slippage=str(cfg_backtest.get("modo_slippage", "fijo")),
        )

    def escalar(self, factor: float) -> Costes:
        """Costes multiplicados (prueba de estrés: x2)."""
        return replace(self, comision=self.comision * factor, slippage=self.slippage * factor)

    def sin_costes(self) -> Costes:
        return replace(self, comision=0.0, slippage=0.0)

    def array_slippage(self, df: pd.DataFrame) -> np.ndarray:
        """Slippage por vela de ejecución. El valor de la vela i usa datos hasta i-1."""
        n = len(df)
        if self.slippage == 0 or self.modo_slippage == "fijo":
            return np.full(n, self.slippage)
        # Se compara con la mediana HISTÓRICA hasta ese momento (expanding), nunca con la de toda la serie
        if self.modo_slippage == "volatilidad":
            medida = (atr(df, 14) / df["close"]).shift(1)
            relativo = medida / medida.expanding(min_periods=20).median()
        elif self.modo_slippage == "volumen":
            vol = df["volume"].replace(0, np.nan).shift(1)
            relativo = np.sqrt(vol.expanding(min_periods=20).median() / vol)
        else:
            raise ValueError(f"Modo de slippage desconocido: {self.modo_slippage}")
        factor = relativo.clip(0.5, 3.0).fillna(1.0).to_numpy()
        return self.slippage * factor
