"""Limpieza de datos, datos sintéticos e importación de CSV."""

import io

import numpy as np
import pandas as pd

from modulos.trading.datos import descarga
from modulos.trading.datos.limpieza import limpiar


def test_limpieza_reporta_sin_inventar():
    idx = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC", name="timestamp")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0}, index=idx)
    df = df.drop(idx[4])  # hueco
    df.loc[idx[6], "close"] = -1  # precio imposible
    df.loc[idx[7], "volume"] = 0
    df.loc[idx[8], "close"] = 200  # salto anómalo
    df = pd.concat([df, df.iloc[[0]]])  # duplicado
    limpio, inf = limpiar(df.reset_index(), "1d", 0.35)
    assert inf["duplicados_eliminados"] == 1
    assert inf["velas_invalidas_eliminadas"] == 1
    assert inf["velas_faltantes"] == 2  # el hueco + la vela inválida eliminada
    assert inf["volumen_cero"] == 1
    assert inf["saltos_anomalos"] >= 1
    assert inf["huecos_rellenados"] == 0
    assert len(limpio) == 8


def test_sinteticos_reproducibles_y_validos():
    a = descarga.generar_sinteticos("ETH/USDT", "1h", "2023-01-01", "2023-03-01", semilla=1)
    b = descarga.generar_sinteticos("ETH/USDT", "1h", "2023-01-01", "2023-03-01", semilla=1)
    pd.testing.assert_frame_equal(a, b)
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1)).all()
    assert (a > 0).all().all()


def test_importar_csv():
    ts = pd.date_range("2024-01-01", periods=50, freq="1D", tz="UTC")
    csv = pd.DataFrame(
        {"timestamp": (ts.astype("int64") // 10**6), "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 3.0}
    ).to_csv(index=False)
    inf = descarga.importar_csv(io.StringIO(csv), "PRUEBA/EUR", "1d")
    assert inf["velas_finales"] == 50
    df = descarga.cargar("PRUEBA/EUR", "1d", "csv")
    assert df.index[0] == ts[0]
    assert np.allclose(df["close"], 1.5)
