"""El motor coincide con casos calculados a mano."""

import numpy as np
import pytest

from modulos.trading.backtest.costes import Costes
from modulos.trading.backtest.motor import Mercado, backtest, backtest_rapido
from tests.conftest import velas


def test_una_compra_y_una_venta_calculada_a_mano():
    # Cierres y aperturas conocidos. Señal: largo al cierre de las velas 0 y 1, fuera después.
    df = velas([100, 105, 115, 125, 135], aperturas=[100, 100, 110, 120, 130])
    obj = np.array([1, 1, 0, 0, 0], dtype=float)
    res = backtest(Mercado(df, "1d"), obj, Costes(comision=0.001, slippage=0.0), 1000.0)

    # Compra en la APERTURA de la vela 1 (100): nocional = 1000/1,001; comisión = nocional·0,001
    nocional = 1000 / 1.001
    unidades = nocional / 100
    # Venta en la apertura de la vela 3 (120) con comisión del 0,1 %
    final = unidades * 120 * (1 - 0.001)
    assert res.equity.iloc[0] == pytest.approx(1000.0)
    assert res.equity.iloc[1] == pytest.approx(unidades * 105)  # valorado a cierre
    assert res.equity.iloc[-1] == pytest.approx(final)
    op = res.operaciones.iloc[0]
    assert len(res.operaciones) == 1
    assert op["precio_entrada"] == 100 and op["precio_salida"] == 120
    assert op["comision"] == pytest.approx(nocional * 0.001 + unidades * 120 * 0.001)
    assert op["resultado"] == pytest.approx(final - 1000)
    assert op["duracion_velas"] == 2
    assert res.metricas["n_ops"] == 1
    assert res.metricas["rentabilidad"] == pytest.approx(final / 1000 - 1)


def test_slippage_empeora_precios():
    df = velas([100, 100, 100, 100], aperturas=[100, 100, 100, 100])
    obj = np.array([1, 0, 0, 0], dtype=float)
    res = backtest(Mercado(df, "1d"), obj, Costes(comision=0.0, slippage=0.01), 1000.0)
    op = res.operaciones.iloc[0]
    assert op["precio_entrada"] == pytest.approx(101.0)  # compras más caro
    assert op["precio_salida"] == pytest.approx(99.0)  # vendes más barato
    assert res.equity.iloc[-1] == pytest.approx(1000 / 101 * 99)


def test_corto_gana_si_baja():
    df = velas([100, 100, 90, 80], aperturas=[100, 100, 90, 80])
    obj = np.array([-1, -1, 0, 0], dtype=float)
    res = backtest(Mercado(df, "1d"), obj, Costes(0.0, 0.0), 1000.0)
    # Corto en 100 (vela 1), recompra en 80 (vela 3): +20 %
    assert res.equity.iloc[-1] == pytest.approx(1200.0)
    assert res.operaciones.iloc[0]["direccion"] == -1


def test_operacion_abierta_se_valora_a_mercado():
    df = velas([100, 110, 120], aperturas=[100, 100, 110])
    res = backtest(Mercado(df, "1d"), np.ones(3), Costes(0.0, 0.0), 1000.0)
    assert bool(res.operaciones.iloc[-1]["abierta"])
    assert res.equity.iloc[-1] == pytest.approx(1000 / 100 * 120)
    assert res.metricas["exposicion"] == pytest.approx(2 / 3)


def test_tramo_de_test_ejecuta_la_senal_previa():
    # Con ini>0, la señal del cierre de ini-1 se ejecuta en la apertura de ini (sin mirar el futuro)
    df = velas([100, 100, 100, 110], aperturas=[100, 100, 100, 100])
    obj = np.array([0, 0, 1, 1], dtype=float)
    res = backtest(Mercado(df, "1d"), obj, Costes(0.0, 0.0), 1000.0, ini=3, fin=4)
    assert res.equity.iloc[0] == pytest.approx(1100.0)


def test_rapido_y_completo_coinciden(df_sintetico):
    from modulos.trading.estrategias import crear

    m = Mercado(df_sintetico, "1d")
    pos = crear("cruce_medias", {"tipo_media": "sma", "rapida": 10, "lenta": 50}).posiciones(df_sintetico)
    c = Costes()
    completo = backtest(m, pos, c, 1000.0)
    rapido = backtest_rapido(m, pos, c.array_slippage(df_sintetico), c.comision, 1000.0, 0, m.n)
    assert list(completo.metricas.values()) == pytest.approx(list(rapido))


def test_sin_costes_nunca_peor_que_con_costes(df_sintetico):
    from modulos.trading.estrategias import crear

    m = Mercado(df_sintetico, "1d")
    pos = crear("rsi_reversion", {"periodo": 14, "compra": 30, "venta": 70}).posiciones(df_sintetico)
    con = backtest(m, pos, Costes(), 1000.0).equity.iloc[-1]
    sin = backtest(m, pos, Costes(0.0, 0.0), 1000.0).equity.iloc[-1]
    assert sin >= con
