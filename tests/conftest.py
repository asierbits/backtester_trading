"""Configuración común de los tests: base de datos y datos en una carpeta temporal."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

_TMP = tempfile.mkdtemp(prefix="laboratorio_tests_")
os.environ["LABORATORIO_DATOS"] = str(Path(_TMP) / "data")
os.environ["LABORATORIO_INFORMES"] = str(Path(_TMP) / "informes")
os.environ.pop("ANTHROPIC_API_KEY", None)


def velas(
    precios: list[float] | np.ndarray, aperturas: list[float] | np.ndarray | None = None, freq: str = "1D"
) -> pd.DataFrame:
    """DataFrame OHLCV sencillo a partir de cierres (y aperturas opcionales)."""
    close = np.asarray(precios, dtype=float)
    open_ = np.asarray(aperturas, dtype=float) if aperturas is not None else np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2022-01-01", periods=len(close), freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": np.full(len(close), 1000.0),
        },
        index=idx,
    )


@pytest.fixture
def df_sintetico() -> pd.DataFrame:
    from modulos.trading.datos.descarga import generar_sinteticos

    return generar_sinteticos("BTC/USDT", "1d", "2020-01-01", "2024-12-31", semilla=7)
