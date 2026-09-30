"""Cambiar datos futuros no puede alterar decisiones ni resultados pasados."""

import numpy as np
import pytest

from modulos.trading.backtest.costes import Costes
from modulos.trading.backtest.motor import Mercado, backtest
from modulos.trading.estrategias import crear

CASOS = [
    ("cruce_medias", {"tipo_media": "sma", "rapida": 10, "lenta": 40}),
    ("cruce_medias", {"tipo_media": "ema", "rapida": 8, "lenta": 30}),
    ("rsi_reversion", {"periodo": 14, "compra": 30, "venta": 70}),
    ("ruptura_donchian", {"entrada": 20, "salida": 10}),
    ("bollinger", {"periodo": 20, "desviaciones": 2.0, "modo": "reversion"}),
    ("bollinger", {"periodo": 20, "desviaciones": 2.0, "modo": "ruptura"}),
    ("macd", {"rapida": 12, "lenta": 26, "senal": 9}),
    ("combinada", {"base": "ruptura_donchian", "filtro": 100, "b_entrada": 20, "b_salida": 10}),
    ("combinada", {"base": "rsi_reversion", "filtro": 150, "b_periodo": 10, "b_compra": 30, "b_venta": 70}),
]


def _alterar_futuro(df, t, rng):
    otro = df.copy()
    ruido = np.exp(rng.normal(0, 0.2, len(df) - t - 1))
    for col in ("open", "high", "low", "close"):
        otro.iloc[t + 1 :, otro.columns.get_loc(col)] = df[col].to_numpy()[t + 1 :] * ruido
    futuro = otro.index[t + 1 :]
    otro.loc[futuro, "high"] = otro.loc[futuro, ["open", "high", "close"]].max(axis=1) * 1.01
    otro.loc[futuro, "low"] = otro.loc[futuro, ["open", "low", "close"]].min(axis=1) * 0.99
    otro.loc[futuro, "volume"] = otro.loc[futuro, "volume"] * rng.uniform(0.2, 5, len(futuro))
    return otro


@pytest.mark.parametrize("cortos", [False, True])
@pytest.mark.parametrize("familia,params", CASOS)
def test_senales_no_miran_el_futuro(df_sintetico, familia, params, cortos):
    rng = np.random.default_rng(1)
    est = crear(familia, params)
    original = est.posiciones(df_sintetico, cortos=cortos)
    for t in (300, 700, 1200):
        alterado = est.posiciones(_alterar_futuro(df_sintetico, t, rng), cortos=cortos)
        np.testing.assert_array_equal(original[: t + 1], alterado[: t + 1])


@pytest.mark.parametrize("modo", ["fijo", "volatilidad", "volumen"])
@pytest.mark.parametrize("familia,params", CASOS)
def test_motor_no_mira_el_futuro(df_sintetico, familia, params, modo):
    rng = np.random.default_rng(2)
    t = 900
    otro = _alterar_futuro(df_sintetico, t, rng)
    est = crear(familia, params)
    c = Costes(modo_slippage=modo)
    a = backtest(Mercado(df_sintetico, "1d"), est.posiciones(df_sintetico), c, 1000.0)
    b = backtest(Mercado(otro, "1d"), est.posiciones(otro), c, 1000.0)
    # La equity hasta el cierre de t es idéntica (la apertura de t+1 aún no se conoce)
    np.testing.assert_allclose(a.equity.to_numpy()[: t + 1], b.equity.to_numpy()[: t + 1])


def test_la_senal_se_ejecuta_en_la_vela_siguiente():
    from tests.conftest import velas

    # Señal en el cierre de la vela 2 → la compra ocurre en la apertura de la vela 3, no antes
    df = velas([100, 100, 100, 200, 200], aperturas=[100, 100, 100, 150, 200])
    obj = np.array([0, 0, 1, 1, 1], dtype=float)
    res = backtest(Mercado(df, "1d"), obj, Costes(0.0, 0.0), 1000.0)
    assert res.operaciones.iloc[0]["precio_entrada"] == 150
    assert res.equity.iloc[2] == 1000.0
