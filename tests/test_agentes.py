"""Panel de agentes con una API simulada (no se gasta dinero ni hace falta conexión)."""

import json
from types import SimpleNamespace

import pytest

from core import db
from core.config import cargar_config
from modulos.trading import repositorio as repo
from modulos.trading.agentes import panel
from modulos.trading.agentes.cliente import ClienteIA, ErrorAgente, PresupuestoAgotado, validar_critica
from modulos.trading.experimento import ejecutar_experimento
from tests.test_experimento import CAMBIOS


class ApiFalsa:
    """Imita client.messages.create / client.beta.messages.create."""

    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.llamadas = []
        self.messages = SimpleNamespace(create=self._crear)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._crear))

    def _crear(self, **kw):
        self.llamadas.append(kw)
        r = self.respuestas.pop(0) if len(self.respuestas) > 1 else self.respuestas[0]
        texto = r(kw) if callable(r) else r
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=texto)],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
            model=kw["model"],
        )


def critica(recomendacion="dudar", **extra):
    def f(kw):
        d = {
            "agente": "x",
            "puntuacion": 4,
            "preocupaciones": ["El Sharpe cae en test"],
            "puntos_fuertes": ["Muchas operaciones"],
            "recomendacion": recomendacion,
            "explicacion": "Texto concreto.",
        }
        if "veredicto_final" in json.dumps(kw["output_config"]["format"]["schema"]):
            d.update(
                {"veredicto_final": extra.get("veredicto_final", "DESCARTADA"), "prueba_adicional": "Probar en 2018."}
            )
        return json.dumps(d)

    return f


def cfg_agentes(**cambios):
    a = cargar_config()["agentes"]
    a.update({"max_reintentos": 2, **cambios})
    return a


def test_validar_critica():
    assert validar_critica(
        {
            "agente": "a",
            "puntuacion": 11,
            "preocupaciones": [],
            "puntos_fuertes": [],
            "recomendacion": "sí",
            "explicacion": "x",
        }
    )
    assert not validar_critica(json.loads(critica()({"output_config": {"format": {"schema": {}}}})))


def test_json_invalido_se_reintenta(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    api = ApiFalsa(["esto no es json", critica()])
    c = ClienteIA(cfg_agentes(), api=api)
    datos, info = c.pedir_json("sistema", "usuario único 1", {"type": "object"}, validar_critica)
    assert datos["recomendacion"] == "dudar"
    assert len(api.llamadas) == 2 and c.uso.errores == 1
    # Fallbacks activados por defecto
    assert api.llamadas[0]["fallbacks"] == "default"


def test_cache_evita_repetir(monkeypatch):
    api = ApiFalsa([critica()])
    c = ClienteIA(cfg_agentes(), api=api)
    c.pedir_json("s", "usuario cache", {"type": "object"}, validar_critica)
    c.pedir_json("s", "usuario cache", {"type": "object"}, validar_critica)
    assert len(api.llamadas) == 1 and c.uso.desde_cache == 1


def test_presupuesto():
    api = ApiFalsa([critica()])
    c = ClienteIA(cfg_agentes(max_llamadas_por_experimento=1), api=api)
    c.pedir_json("s", "u1", {"type": "object"}, validar_critica)
    with pytest.raises(PresupuestoAgotado):
        c.pedir_json("s", "u2", {"type": "object"}, validar_critica)


def test_sin_respuesta_valida_lanza_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    c = ClienteIA(cfg_agentes(), api=ApiFalsa(["{}"]))
    with pytest.raises(ErrorAgente):
        c.pedir_json("s", "u-malo", {"type": "object"}, validar_critica)


def test_panel_completo_endurece_y_nunca_relaja():
    exp_id = ejecutar_experimento(CAMBIOS, "agentes", lanzar_agentes=False)
    ids = repo.finalistas(exp_id)
    # Todos los agentes piden aprobar y el árbitro dice APROBADA: no puede relajar una DESCARTADA/SOSPECHOSA
    api = ApiFalsa([critica("aprobar", veredicto_final="APROBADA")])
    antes = {i: repo.estrategia(i)["veredicto"] for i in ids}
    info = panel.ejecutar_panel(exp_id, ids=ids, api=api)
    assert info["evaluadas"] == len(ids)
    for i in ids:
        e = repo.estrategia(i)
        assert e["veredicto"] == antes[i] or antes[i] == "APROBADA"
        assert len(e["criticas"]) == len(cargar_config()["agentes"]["panel"]) + 1
    # Ahora el árbitro descarta: todas pasan a DESCARTADA. (Con la misma entrada saldría de la
    # caché, así que se cambia el esfuerzo, que forma parte de la clave de caché.)
    exp2 = ejecutar_experimento(CAMBIOS, "agentes 2", lanzar_agentes=False)
    cfg2 = db.obtener_experimento(exp2)["config"]
    cfg2["agentes"]["esfuerzo"] = "high"
    api2 = ApiFalsa([critica("descartar", veredicto_final="DESCARTADA")])
    panel.ejecutar_panel(exp2, cfg=cfg2, api=api2)
    assert api2.llamadas
    assert all(repo.estrategia(i)["veredicto"] == "DESCARTADA" for i in repo.finalistas(exp2))


def test_resumen_para_agentes_es_compacto():
    exp_id = ejecutar_experimento(CAMBIOS, "resumen", lanzar_agentes=False)
    r = panel.resumen_para_agentes(repo.finalistas(exp_id)[0])
    texto = json.dumps(r)
    assert len(texto) < 20_000
    assert "trayectorias" not in texto and "abanico" not in texto
    assert r["mercado"]["datos_sinteticos"] is True
    est = panel.estimar_coste_panel(exp_id, cargar_config(), repo.finalistas(exp_id))
    assert est["llamadas"] == 3 * 6 and est["coste_usd"] > 0
