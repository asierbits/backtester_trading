"""Flujo completo con datos sintéticos: experimento, reproducibilidad, validación e informe."""

import numpy as np
import pytest

from core import db
from modulos.trading import informes
from modulos.trading import repositorio as repo
from modulos.trading.experimento import ejecutar_experimento, validar_a_fondo

CAMBIOS = {
    "general": {"nucleos": 1},
    "datos": {"desde": "2020-01-01", "hasta": "2024-06-30"},
    "experimento": {
        "fuente": "sintetico",
        "temporalidad": "1d",
        "activos": ["BTC/USDT", "ETH/USDT"],
        "max_estrategias": 120,
        "familias": ["cruce_medias", "rsi_reversion", "ruptura_donchian", "combinada"],
        "n_aleatorias": 40,
        "top_finalistas": 3,
    },
    "validacion": {"walk_forward_candidatos": 20, "monte_carlo_simulaciones": 200},
}


@pytest.fixture(scope="module")
def dos_experimentos():
    a = ejecutar_experimento(CAMBIOS, "prueba A", lanzar_agentes=False)
    b = ejecutar_experimento(CAMBIOS, "prueba B", lanzar_agentes=False)
    return a, b


def test_experimento_completo(dos_experimentos):
    a, _ = dos_experimentos
    exp = db.obtener_experimento(a)
    r = exp["resumen"]
    assert exp["estado"] == "terminado"
    assert r["n_estrategias"] == 240 and r["n_combinaciones"] == 120
    assert sum(r["conteo"].values()) == 240
    assert r["datos_sinteticos"] is True
    rk = repo.ranking(a)
    assert len(rk) == 240
    assert set(rk["veredicto"]) <= {"APROBADA", "SOSPECHOSA", "DESCARTADA"}
    fin = repo.finalistas(a)
    assert len(fin) == 3
    # Las finalistas se eligieron por Sharpe de ENTRENAMIENTO
    elegibles = rk[rk["ops_train"] >= exp["config"]["veredicto"]["min_operaciones"]]
    top = elegibles.sort_values("sharpe_train", ascending=False)["id"].head(3).tolist()
    assert sorted(top) == sorted(fin)
    info = repo.estrategia(fin[0])
    for tipo in (
        "train_test",
        "walk_forward",
        "monte_carlo",
        "robustez",
        "multiples_pruebas",
        "doble_coste",
        "regimenes",
        "aleatoria",
        "buy_hold",
        "min_operaciones",
    ):
        assert tipo in info["validaciones"], tipo
    assert info["motivos"]
    assert "test_x2" in info["resultados"]
    assert info["resultados"]["test"]["rentabilidad_bruta"] is not None
    # La configuración exacta y la semilla quedan guardadas
    assert exp["config"]["experimento"]["max_estrategias"] == 120
    assert exp["semilla"] == 42


def test_reproducible(dos_experimentos):
    a, b = dos_experimentos
    ra, rb = repo.ranking(a), repo.ranking(b)
    cols = ["familia", "activo", "sharpe_train", "sharpe_test", "rent_test", "veredicto"]
    ra = ra.sort_values(["activo", "familia", "parametros_json"])[cols + ["parametros_json"]].reset_index(drop=True)
    rb = rb.sort_values(["activo", "familia", "parametros_json"])[cols + ["parametros_json"]].reset_index(drop=True)
    assert ra.equals(rb)
    va = repo.estrategia(repo.finalistas(a)[0])["validaciones"]["monte_carlo"]["rentabilidad_p50"]
    vb = repo.estrategia(repo.finalistas(b)[0])["validaciones"]["monte_carlo"]["rentabilidad_p50"]
    assert va == vb


def test_referencias_de_control(dos_experimentos):
    a, _ = dos_experimentos
    refs = repo.referencias(a)
    assert set(refs["tipo"]) == {"buy_hold", "aleatoria"}
    al = refs[refs["tipo"] == "aleatoria"].iloc[0]["datos"]
    assert len(al["sharpe"]) == 40
    assert np.isfinite(al["p95_sharpe"])


def test_validar_a_fondo_una_no_finalista(dos_experimentos):
    a, _ = dos_experimentos
    rk = repo.ranking(a)
    otra = int(rk[rk["finalista"] == 0].iloc[0]["id"])
    v, motivos = validar_a_fondo(otra)
    assert v in {"APROBADA", "SOSPECHOSA", "DESCARTADA"}
    assert "walk_forward" in repo.estrategia(otra)["validaciones"]


def test_informe_html(dos_experimentos):
    a, _ = dos_experimentos
    html = informes.generar_html(a, n_detalle=2)
    assert "Laboratorio de humo" in html and "SINTÉTICOS" in html
    assert "consejo financiero" in html
    ruta = informes.guardar(a, 1)
    assert ruta.exists() and ruta.stat().st_size > 1000


def test_cartera(dos_experimentos):
    from modulos.trading.paper import cartera

    a, _ = dos_experimentos
    ids = repo.finalistas(a)[:2]
    c = cartera.construir_cartera(ids, "minima_varianza")
    assert c["pesos"].sum() == pytest.approx(1.0)
    assert (c["pesos"] >= 0).all()
    assert "sharpe" in c["metricas"]
