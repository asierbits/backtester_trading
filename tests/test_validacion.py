"""Validación anti-humo: estadística, Monte Carlo, walk-forward y reglas del veredicto."""

import numpy as np
import pytest

from core.config import cargar_config
from modulos.trading.validacion import monte_carlo, multiples_pruebas, veredicto, walk_forward


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    rech = multiples_pruebas.benjamini_hochberg(p, 0.05)
    assert rech.tolist() == [True, True] + [False] * 8
    assert multiples_pruebas.bonferroni(p, 0.05).sum() == 1


def test_mas_pruebas_exigen_mas_sharpe():
    a = multiples_pruebas.sharpe_maximo_esperado(10, 0.01)
    b = multiples_pruebas.sharpe_maximo_esperado(10_000, 0.01)
    assert 0 < a < b


def test_dsr_penaliza_el_numero_de_pruebas():
    sr = 0.08  # por vela
    pocas, _ = multiples_pruebas.deflated_sharpe(sr, 1000, 0, 3, 2, 0.001)
    muchas, _ = multiples_pruebas.deflated_sharpe(sr, 1000, 0, 3, 100_000, 0.001)
    assert muchas < pocas
    assert multiples_pruebas.sharpe_probabilistico(0.1, 1000) > 0.99


def test_reality_check():
    rng = np.random.default_rng(0)
    azar = rng.normal(0, 0.5, 500)
    assert multiples_pruebas.reality_check(5.0, azar, 1000, 500)["supera"]
    assert not multiples_pruebas.reality_check(1.0, azar, 1000, 500)["supera"]


def test_bootstrap_operaciones():
    r = np.array([0.05, -0.02, 0.03, -0.01, 0.04] * 10)
    mc = monte_carlo.bootstrap_operaciones(r, 500, 1)
    assert mc["valido"]
    assert mc["rentabilidad_p5"] <= mc["rentabilidad_p50"] <= mc["rentabilidad_p95"]
    assert 0 <= mc["prob_perdida"] <= 1
    assert len(mc["abanico"]["50"]) == len(r)
    assert not monte_carlo.bootstrap_operaciones(np.array([0.1]), 10, 1)["valido"]


def test_vecinos_de_parametros():
    rng = np.random.default_rng(0)
    v = monte_carlo.vecinos(
        {"rapida": 20, "lenta": 100, "tipo_media": "sma"},
        {"rapida": [5, 50, 1], "lenta": [20, 250, 5], "tipo_media": ["sma", "ema"]},
        5,
        rng,
    )
    assert any(x["rapida"] == 22 for x in v)  # 20 ± 10 %
    assert all(x["tipo_media"] == "sma" for x in v)


def test_ventanas_walk_forward():
    v = walk_forward.limites_ventanas(700, 5)
    assert len(v) == 5
    for ini_opt, ini_test, fin_test in v:
        assert ini_opt < ini_test < fin_test
    assert v[-1][2] == 700


def _validaciones_buenas():
    return {
        "min_operaciones": {"train": 80, "test": 40, "supera": True},
        "train_test": {"sharpe_train": 1.5, "sharpe_test": 1.2, "conservacion": 0.8, "supera": True},
        "aleatoria": {"sharpe_test": 1.2, "umbral_test": 0.8, "supera": True},
        "buy_hold": {"sharpe": 1.2, "sharpe_bh": 0.9, "calmar": 1.0, "calmar_bh": 0.5, "supera": True},
        "doble_coste": {"multiplicador": 2.0, "sharpe": 0.9, "rentabilidad": 0.3, "supera": True},
        "robustez": {"valido": True, "frac_positivas": 0.9, "ratio_mediana": 0.8, "n_vecinos": 20, "supera": True},
        "walk_forward": {"pct_ventanas_rentables": 0.8, "rentabilidad_oos_total": 0.5, "supera": True},
        "monte_carlo": {
            "valido": True,
            "rentabilidad_p5": 0.01,
            "rentabilidad_p95": 0.9,
            "max_dd_p95": 0.3,
            "prob_perdida": 0.04,
            "supera": True,
        },
        "multiples_pruebas": {"explicacion": "DSR alto.", "supera": True},
        "regimenes": {"supera": True, "fallos": [], "solo_gana_en_alcista": False},
    }


def test_veredicto_aprobada_solo_si_todo_pasa():
    cfg = cargar_config()
    v, motivos = veredicto.veredicto_completo(_validaciones_buenas(), cfg)
    assert v == veredicto.APROBADA
    val = _validaciones_buenas()
    val["walk_forward"]["supera"] = False
    assert veredicto.veredicto_completo(val, cfg)[0] == veredicto.SOSPECHOSA


def test_veredicto_descarta_si_falla_fuera_de_muestra():
    cfg = cargar_config()
    val = _validaciones_buenas()
    val["train_test"].update({"sharpe_test": -0.1, "supera": False})
    v, motivos = veredicto.veredicto_completo(val, cfg)
    assert v == veredicto.DESCARTADA
    assert "1,50" in motivos[0] and "-0,10" in motivos[0]


@pytest.mark.parametrize("fallo", ["robustez", "doble_coste", "min_operaciones"])
def test_fallos_graves_descartan(fallo):
    val = _validaciones_buenas()
    val[fallo]["supera"] = False
    assert veredicto.veredicto_completo(val, cargar_config())[0] == veredicto.DESCARTADA


def test_veredicto_basico():
    cfg = cargar_config()
    buena = {"n_ops": 50, "sharpe": 1.5}
    assert veredicto.veredicto_basico(buena, {"n_ops": 20, "sharpe": 1.0}, 1.0, cfg)[0] == veredicto.SOSPECHOSA
    assert (
        veredicto.veredicto_basico({"n_ops": 5, "sharpe": 3.0}, {"n_ops": 20, "sharpe": 1.0}, 1.0, cfg)[0]
        == veredicto.DESCARTADA
    )
    assert veredicto.veredicto_basico(buena, {"n_ops": 20, "sharpe": 1.0}, 2.0, cfg)[0] == veredicto.DESCARTADA


def test_agentes_solo_endurecen():
    A, S, D = veredicto.APROBADA, veredicto.SOSPECHOSA, veredicto.DESCARTADA
    assert veredicto.endurecer(D, "aprobar") == D
    assert veredicto.endurecer(S, "aprobar") == S
    assert veredicto.endurecer(A, "dudar") == S
    assert veredicto.endurecer(A, "descartar") == D
    assert veredicto.endurecer(S, None) == S
