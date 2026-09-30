"""Generador de variaciones: tamaño, validez y reproducibilidad."""

from core.config import cargar_config
from modulos.trading.estrategias import FAMILIAS
from modulos.trading.estrategias.generador import generar, repartir_cuotas


def test_reproducible_y_valido():
    p = cargar_config()["parametros"]
    familias = ["cruce_medias", "rsi_reversion", "ruptura_donchian", "combinada"]
    a = generar(familias, p, 2000, 42)
    b = generar(familias, p, 2000, 42)
    c = generar(familias, p, 2000, 43)
    assert a.combinaciones == b.combinaciones
    assert a.combinaciones != c.combinaciones
    assert a.n == 2000
    assert a.espacio_total > 2000
    for f, params in a.combinaciones:
        if f == "combinada":
            base = {k[2:]: v for k, v in params.items() if k.startswith("b_")}
            assert FAMILIAS[params["base"]].valida(base)
        else:
            assert FAMILIAS[f].valida(params)
    # Sin duplicados
    assert len({(f, tuple(sorted(p.items()))) for f, p in a.combinaciones}) == a.n


def test_familia_pequena_se_prueba_entera():
    cuotas = repartir_cuotas({"a": 10, "b": 10_000}, 1000)
    assert cuotas == {"a": 10, "b": 990}


def test_cruce_medias_rapida_menor_que_lenta():
    p = cargar_config()["parametros"]
    for combo in FAMILIAS["cruce_medias"].rejilla(p["cruce_medias"]):
        assert combo["rapida"] < combo["lenta"]
