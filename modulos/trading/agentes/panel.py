"""Orquesta el panel de agentes críticos sobre las estrategias finalistas.

Para cada finalista:
1. Se construye un resumen estructurado en JSON (métricas, validaciones, veredicto
   automático y curvas muy resumidas; nunca datos crudos enormes).
2. Los agentes críticos lo evalúan en paralelo (con límite de concurrencia).
3. El Árbitro lee las críticas y el veredicto automático y redacta el veredicto final.
4. El veredicto final solo puede ser igual o MÁS DURO que el automático.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from core import db
from core.logging_setup import obtener_logger
from modulos.trading import repositorio as repo
from modulos.trading.agentes import prompts
from modulos.trading.agentes.cliente import ClienteIA, ErrorAgente, PresupuestoAgotado, validar_critica
from modulos.trading.estrategias import crear
from modulos.trading.validacion import veredicto as ver

log = obtener_logger(__name__)

METRICAS_CLAVE = [
    "rentabilidad",
    "cagr",
    "volatilidad",
    "sharpe",
    "sortino",
    "max_dd",
    "dur_max_dd_dias",
    "calmar",
    "n_ops",
    "pct_ganadoras",
    "ratio_ganancia_perdida",
    "profit_factor",
    "expectativa",
    "exposicion",
    "racha_perdidas",
    "rentabilidad_bruta",
]


def _redondear(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 4) if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _redondear(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_redondear(v) for v in x]
    return x


def resumen_para_agentes(estrategia_id: int) -> dict:
    """Resumen compacto y determinista de una estrategia (entrada de los agentes)."""
    info = repo.estrategia(estrategia_id)
    if info is None:
        raise KeyError(estrategia_id)
    exp = db.obtener_experimento(int(info["experimento_id"]))
    cfg, res_exp = exp["config"], exp["resumen"]
    activo = res_exp["activos"][info["activo"]]
    val = json.loads(json.dumps(info.get("validaciones", {})))  # copia profunda

    # Quitar arrays grandes y dejar solo lo que un humano miraría
    if "walk_forward" in val:
        wf = val["walk_forward"]
        curva = wf.pop("curva", {})
        eq = curva.get("equity", [])
        wf["curva_oos_resumida"] = [round(x, 1) for x in eq[:: max(1, len(eq) // 12)]]
    if "monte_carlo" in val:
        val["monte_carlo"].pop("abanico", None)
        val["monte_carlo"].pop("trayectorias", None)
    if "robustez" in val:
        vecinos = val["robustez"].pop("vecinos", [])
        val["robustez"]["peores_vecinos"] = sorted(vecinos, key=lambda v: v["sharpe"])[:3]

    ops = repo.operaciones(estrategia_id, "test")
    ops_resumen = {}
    if len(ops):
        ops_resumen = {
            "n": int(len(ops)),
            "mejor_operacion_pct": float(ops["resultado_pct"].max()),
            "peor_operacion_pct": float(ops["resultado_pct"].min()),
            "duracion_media_velas": float(ops["duracion_velas"].mean()),
            "pct_beneficio_de_la_mejor_operacion": float(
                ops["resultado"].max() / ops.loc[ops["resultado"] > 0, "resultado"].sum()
            )
            if (ops["resultado"] > 0).any()
            else None,
        }
    anos_test = (activo["velas"] - activo["corte"]) / activo["velas_ano"]
    estrategia = crear(info["familia"], info["parametros"])
    resultados = {
        tramo: {k: v for k, v in m.items() if k in METRICAS_CLAVE}
        for tramo, m in info.get("resultados", {}).items()
        if tramo in ("train", "test", "test_x2")
    }
    return _redondear(
        {
            "estrategia": {
                "familia": info["familia"],
                "descripcion": estrategia.descripcion,
                "nombre": estrategia.descripcion_corta(),
                "parametros": info["parametros"],
                "n_parametros": len(info["parametros"]),
            },
            "mercado": {
                "activo": info["activo"],
                "temporalidad": info["temporalidad"],
                "fuente_datos": res_exp.get("fuente"),
                "datos_sinteticos": bool(res_exp.get("datos_sinteticos")),
                "periodo_train": [activo["desde"][:10], activo["inicio_test"][:10]],
                "periodo_test": [activo["inicio_test"][:10], activo["hasta"][:10]],
            },
            "condiciones_backtest": {
                "capital_inicial_eur": cfg["backtest"]["capital_inicial"],
                "comision": cfg["backtest"]["comision"],
                "slippage": cfg["backtest"]["slippage"],
                "ejecucion": "señal al cierre de t, ejecución en la vela t+1 ("
                + cfg["backtest"].get("precio_ejecucion", "apertura")
                + ")",
                "tamano_posicion": "100 % del capital, sin apalancamiento"
                if cfg["backtest"].get("sizing", "completo") == "completo"
                else "sizing por volatilidad",
                "cortos_permitidos": bool(cfg["backtest"].get("permitir_cortos")),
            },
            "seleccion": {
                "estrategias_probadas_en_el_experimento": res_exp.get("n_estrategias"),
                "rango_por_sharpe_de_entrenamiento": info.get("rango_train"),
                "nota": "Se eligió solo con el tramo de entrenamiento; el test no se tocó.",
            },
            "metricas": resultados,
            "operaciones_test": ops_resumen,
            "operaciones_por_ano_test": (resultados.get("test", {}).get("n_ops", 0) / anos_test) if anos_test else None,
            "comprar_y_mantener": {"train": activo["bh_train"], "test": activo["bh_test"]},
            "validaciones": val,
            "veredicto_automatico": {
                "veredicto": info.get("veredicto_automatico"),
                "motivos": info.get("motivos", []),
            },
        }
    )


def _consenso(criticas: list[dict]) -> str | None:
    """Recomendación mayoritaria (en empate, la más dura)."""
    if not criticas:
        return None
    votos = Counter(c["recomendacion"] for c in criticas)
    orden = {"aprobar": 0, "dudar": 1, "descartar": 2}
    return max(votos, key=lambda r: (votos[r], orden[r]))


def evaluar_estrategia(
    cliente: ClienteIA,
    estrategia_id: int,
    agentes: list[str],
    usar_arbitro: bool,
    ejecutor: ThreadPoolExecutor,
) -> dict:
    """Pasa una estrategia por el panel y actualiza su veredicto final."""
    resumen = resumen_para_agentes(estrategia_id)

    def critica(clave: str) -> dict | None:
        try:
            datos, info = cliente.pedir_json(
                prompts.sistema(clave),
                prompts.mensaje_agente(clave, resumen),
                prompts.ESQUEMA_CRITICA,
                validar_critica,
            )
        except PresupuestoAgotado:
            raise
        except ErrorAgente as e:
            log.error("Agente %s falló en la estrategia %s: %s", clave, estrategia_id, e)
            return None
        datos["agente"] = prompts.nombre(clave)
        repo.guardar_critica(
            estrategia_id,
            clave,
            prompts.VERSION_PROMPTS,
            info["hash"],
            info.get("modelo", cliente.modelo),
            datos,
            int(info.get("tokens_entrada") or 0),
            int(info.get("tokens_salida") or 0),
        )
        return datos

    criticas = [c for c in ejecutor.map(critica, agentes) if c is not None]

    arbitro = None
    if usar_arbitro and criticas:
        try:
            arbitro, info = cliente.pedir_json(
                prompts.sistema("arbitro"),
                prompts.mensaje_arbitro(resumen, criticas),
                prompts.ESQUEMA_ARBITRO,
                lambda d: validar_critica(d, arbitro=True),
            )
            arbitro["agente"] = prompts.nombre("arbitro")
            repo.guardar_critica(
                estrategia_id,
                "arbitro",
                prompts.VERSION_PROMPTS,
                info["hash"],
                info.get("modelo", cliente.modelo),
                arbitro,
                int(info.get("tokens_entrada") or 0),
                int(info.get("tokens_salida") or 0),
            )
        except PresupuestoAgotado:
            raise
        except ErrorAgente as e:
            log.error("El árbitro falló en la estrategia %s: %s", estrategia_id, e)

    automatico = resumen["veredicto_automatico"]["veredicto"] or ver.SOSPECHOSA
    motivos = list(resumen["veredicto_automatico"]["motivos"])
    if arbitro:
        # El árbitro decide; si su veredicto y su recomendación discrepan, gana el más duro
        a_recomendacion = {ver.APROBADA: "aprobar", ver.SOSPECHOSA: "dudar", ver.DESCARTADA: "descartar"}
        opciones = [a_recomendacion[arbitro["veredicto_final"]], arbitro["recomendacion"]]
        recomendacion = max(opciones, key=lambda r: ver.GRAVEDAD[ver.DESDE_RECOMENDACION[r]])
    else:
        recomendacion = _consenso(criticas)
    final = ver.endurecer(automatico, recomendacion)
    if criticas:
        texto = arbitro["explicacion"] if arbitro else "; ".join(c["explicacion"] for c in criticas[:2])
        if final != automatico:
            motivos.insert(0, f"✗ Endurecida por el panel de agentes ({automatico} → {final}): {texto}")
        else:
            motivos.append(f"• Panel de agentes: {texto}")
        repo.actualizar_veredicto_final(estrategia_id, final, motivos)
    return {
        "estrategia_id": estrategia_id,
        "automatico": automatico,
        "final": final,
        "n_criticas": len(criticas),
        "arbitro": bool(arbitro),
    }


def estimar_coste_panel(experimento_id: int, cfg: dict, ids: list[int] | None = None) -> dict:
    """Estimación del coste de pasar las finalistas por el panel, antes de lanzarlo."""
    ids = ids if ids is not None else repo.finalistas(experimento_id)
    cliente = ClienteIA(cfg["agentes"], api=object())
    agentes = list(cfg["agentes"]["panel"])
    entradas: list[str] = []
    for i in ids:
        resumen = resumen_para_agentes(i)
        entradas += [prompts.sistema(a) + prompts.mensaje_agente(a, resumen) for a in agentes]
        if cfg["agentes"].get("arbitro", True):
            falsas = [{"agente": a, "explicacion": "x" * 600, "preocupaciones": ["x" * 150] * 4} for a in agentes]
            entradas.append(prompts.sistema("arbitro") + prompts.mensaje_arbitro(resumen, falsas))
    estimacion = cliente.estimar_coste(entradas)
    estimacion["n_estrategias"] = len(ids)
    estimacion["dentro_de_limites"] = bool(
        estimacion["llamadas"] <= cfg["agentes"]["max_llamadas_por_experimento"]
        and estimacion["tokens_entrada"] + estimacion["tokens_salida"] <= cfg["agentes"]["max_tokens_por_experimento"]
    )
    return estimacion


def ejecutar_panel(
    experimento_id: int,
    cfg: dict | None = None,
    ids: list[int] | None = None,
    progreso: Callable[[float, str], None] | None = None,
    api: Any | None = None,
) -> dict:
    """Pasa las finalistas (o `ids`) por el panel. Devuelve consumo y resultados."""
    cfg = cfg or db.obtener_experimento(experimento_id)["config"]
    ids = ids if ids is not None else repo.finalistas(experimento_id)
    ca = cfg["agentes"]
    cliente = ClienteIA(ca, api=api)
    estimacion = estimar_coste_panel(experimento_id, cfg, ids)
    log.info(
        "Panel de agentes: %s estrategias, ~%s llamadas, coste estimado ~%.2f $",
        len(ids),
        estimacion["llamadas"],
        estimacion["coste_usd"],
    )
    resultados, aviso = [], None
    with ThreadPoolExecutor(max_workers=max(1, int(ca.get("concurrencia", 3)))) as ejecutor:
        for k, est_id in enumerate(ids):
            if progreso:
                progreso(k / max(len(ids), 1), f"Agentes: estrategia {k + 1}/{len(ids)}")
            try:
                resultados.append(
                    evaluar_estrategia(cliente, est_id, list(ca["panel"]), bool(ca.get("arbitro", True)), ejecutor)
                )
            except PresupuestoAgotado as e:
                aviso = f"Presupuesto agotado: {e}. Las estrategias restantes no se evaluaron."
                log.warning(aviso)
                break
    if progreso:
        progreso(1.0, "Panel de agentes terminado")
    u = cliente.uso
    return {
        "estimacion": estimacion,
        "llamadas": u.llamadas,
        "desde_cache": u.desde_cache,
        "tokens_entrada": u.tokens_entrada,
        "tokens_salida": u.tokens_salida,
        "errores": u.errores,
        "coste_usd": cliente.coste_actual(),
        "modelo": ca["modelo"],
        "evaluadas": len(resultados),
        "endurecidas": sum(r["final"] != r["automatico"] for r in resultados),
        "aviso": aviso,
    }
