"""Indicadores contra cálculos de referencia."""

import numpy as np
import pandas as pd
import pytest

from modulos.trading.estrategias import indicadores as ind


def test_sma_y_ema():
    x = pd.Series([1.0, 2, 3, 4, 5])
    np.testing.assert_allclose(ind.sma(x, 3).to_numpy()[2:], [2, 3, 4])
    assert np.isnan(ind.sma(x, 3).iloc[1])
    e = ind.ema(x, 3)  # alfa = 0,5
    assert e.iloc[2] == pytest.approx((1 * 0.5 + 2 * 0.5) * 0.5 + 3 * 0.5)


def test_rsi_extremos():
    subida = pd.Series(np.arange(1, 40, dtype=float))
    assert ind.rsi(subida, 14).iloc[-1] == pytest.approx(100.0)
    bajada = pd.Series(np.arange(40, 1, -1, dtype=float))
    assert ind.rsi(bajada, 14).iloc[-1] == pytest.approx(0.0)


def test_donchian_excluye_la_vela_actual():
    alto = pd.Series([1.0, 2, 3, 10, 4])
    d = ind.donchian_alto(alto, 2)
    assert d.iloc[3] == 3  # máximo de las 2 velas anteriores, no incluye el 10
    assert d.iloc[4] == 10


def test_maquina_estados():
    ent = np.array([False, True, False, False, True])
    sal = np.array([False, False, False, True, False])
    np.testing.assert_array_equal(ind.maquina_estados(ent, sal), [0, 1, 1, 0, 1])
