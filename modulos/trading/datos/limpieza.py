"""Limpieza y control de calidad de velas OHLCV.

Regla de oro: nunca inventamos datos. Los huecos se detectan y se reportan,
pero no se rellenan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNAS = ["open", "high", "low", "close", "volume"]

FRECUENCIAS = {"1h": "1h", "4h": "4h", "1d": "1D"}

VELAS_POR_ANO = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}


def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Índice DatetimeIndex en UTC llamado 'timestamp', columnas OHLCV float."""
    df = df.copy()
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        if np.issubdtype(ts.dtype, np.number):
            # Milisegundos (ccxt) o segundos (algunos CSV)
            unidad = "ms" if ts.max() > 1e11 else "s"
            df["timestamp"] = pd.to_datetime(ts, unit=unidad, utc=True)
        else:
            df["timestamp"] = pd.to_datetime(ts, utc=True)
        df = df.set_index("timestamp")
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas: {faltan}. Se esperan {COLUMNAS}.")
    return df[COLUMNAS].astype(float)


def limpiar(
    df: pd.DataFrame, temporalidad: str, salto_anomalo: float = 0.35
) -> tuple[pd.DataFrame, dict]:
    """Limpia las velas y devuelve (datos_limpios, informe_de_calidad).

    - Elimina duplicados (se queda con la última versión de cada vela).
    - Elimina velas con precios <= 0 o con high < low (datos imposibles).
    - Reporta velas faltantes, volumen cero y saltos anómalos sin tocarlos.
    """
    df = normalizar(df)
    informe: dict = {"velas_originales": int(len(df))}

    duplicados = int(df.index.duplicated(keep="last").sum())
    df = df[~df.index.duplicated(keep="last")].sort_index()
    informe["duplicados_eliminados"] = duplicados

    precios = df[["open", "high", "low", "close"]]
    invalidas = (precios <= 0).any(axis=1) | (df["high"] < df["low"]) | precios.isna().any(axis=1)
    informe["velas_invalidas_eliminadas"] = int(invalidas.sum())
    df = df[~invalidas]

    informe["volumen_cero"] = int((df["volume"] <= 0).sum())

    rent = df["close"].pct_change().abs()
    saltos = rent[rent > salto_anomalo]
    informe["saltos_anomalos"] = int(len(saltos))
    informe["fechas_saltos"] = [str(t) for t in saltos.index[:20]]

    freq = FRECUENCIAS.get(temporalidad)
    if freq and len(df) > 1:
        esperado = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
        faltantes = esperado.difference(df.index)
        informe["velas_faltantes"] = int(len(faltantes))
        informe["pct_faltantes"] = round(100 * len(faltantes) / max(len(esperado), 1), 3)
        informe["primeras_faltantes"] = [str(t) for t in faltantes[:20]]
    else:
        informe["velas_faltantes"] = 0
        informe["pct_faltantes"] = 0.0
        informe["primeras_faltantes"] = []

    informe["huecos_rellenados"] = 0  # Nunca rellenamos: queda constancia explícita
    informe["velas_finales"] = int(len(df))
    return df, informe
