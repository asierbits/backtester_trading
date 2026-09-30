"""Métricas sobre curvas sencillas."""

import numpy as np
import pytest

from modulos.trading.backtest.metricas import CLAVES_METRICAS, formatear, resumen_numba

I = {k: i for i, k in enumerate(CLAVES_METRICAS)}


def test_caida_maxima_y_duracion():
    eq = np.array([100, 120, 90, 60, 80, 130, 125.0])
    m = resumen_numba(eq, 100.0, np.array([0.3, -0.1]), 2, 5, 365.0)
    assert m[I["max_dd"]] == pytest.approx(0.5)  # de 120 a 60
    assert m[I["dur_max_dd_dias"]] == pytest.approx(3.0)  # 3 velas por debajo del máximo
    assert m[I["rentabilidad"]] == pytest.approx(0.25)
    assert m[I["pct_ganadoras"]] == pytest.approx(0.5)
    assert m[I["profit_factor"]] == pytest.approx(3.0)
    assert m[I["exposicion"]] == pytest.approx(5 / 7)


def test_sharpe_positivo_con_subida_constante_con_ruido():
    rng = np.random.default_rng(0)
    r = 0.001 + rng.normal(0, 0.01, 1000)
    eq = 100 * np.cumprod(1 + r)
    m = resumen_numba(eq, 100.0, np.zeros(0), 0, 0, 365.0)
    esperado = r.mean() / r.std() * np.sqrt(365)
    assert m[I["sharpe"]] == pytest.approx(esperado, rel=1e-6)


def test_racha_de_perdidas():
    m = resumen_numba(np.array([100.0]), 100.0, np.array([0.1, -0.1, -0.2, -0.05, 0.3, -0.1]), 6, 0, 365.0)
    assert m[I["racha_perdidas"]] == 3


def test_formato_espanol():
    assert formatear("rentabilidad", 0.1234) == "12,3 %"
    assert formatear("sharpe", 1234.5) == "1.234,50"
