"""Orquestación de un experimento completo.

Pasos:
1. Carga los datos de cada activo y los divide en entrenamiento / test (cronológico).
2. Genera miles de combinaciones de parámetros (y guarda cuántas se probaron).
3. Backtest de cada combinación en cada activo, en paralelo (train y test).
4. Referencias de control: comprar y mantener y estrategias aleatorias.
5. Veredicto básico para todas; selección de finalistas SOLO con el entrenamiento.
6. Validación completa de las finalistas: walk-forward, Monte Carlo, robustez,
   múltiples pruebas, doble coste, regímenes, azar y buy & hold → veredicto.
7. (Opcional) Panel de agentes de IA que critica a las finalistas.

Uso sin interfaz:  python -m modulos.trading.experimento --fuente sintetico
"""

from __future__ import annotations

import math
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from core import db
from core.config import cargar_config, clave_anthropic, fusionar
from core.logging_setup import obtener_logger
from modulos.trading import repositorio as repo
from modulos.trading.backtest.costes import Costes
from modulos.trading.backtest.metricas import CLAVES_METRICAS, decimal, miles
from modulos.trading.backtest.motor import (
    Mercado,
    aplicar_sizing_volatilidad,
    backtest,
    backtest_rapido,
    metricas_a_dict,
)
from modulos.trading.datos.descarga import cargar_o_generar
from modulos.trading.estrategias import FAMILIAS, crear, hash_estrategia
from modulos.trading.estrategias.control import Aleatoria
from modulos.trading.estrategias.generador import generar
from modulos.trading.estrategias.indicadores import CacheIndicadores
from modulos.trading.validacion import (
    monte_carlo,
    multiples_pruebas,
    regimenes,
    train_test,
    veredicto,
    walk_forward,
)

log = obtener_logger(__name__)

Progreso = Callable[[float, str], None]
I = {k: i for i, k in enumerate(CLAVES_METRICAS)}  # noqa: E741


# ============================================================ contexto por activo
@dataclass
class ContextoActivo:
    """Todo lo necesario para hacer backtests de un activo."""

    activo: str
    mercado: Mercado
    corte: int
    slip: np.ndarray
    costes: Costes
    capital: float
    cache: CacheIndicadores = field(repr=False)

    @property
    def n(self) -> int:
        return self.mercado.n


def config_experimento(cambios: dict | None = None) -> dict:
    """Configuración del experimento: config.yaml + cambios de la interfaz."""
    return fusionar(cargar_config(), cambios or {})


def cargar_datos(activo: str, cfg: dict) -> pd.DataFrame:
    """Velas del activo según la configuración del experimento."""
    e = cfg["experimento"]
    df = cargar_o_generar(activo, e["temporalidad"], e["fuente"], cfg["general"]["semilla"])
    if e.get("desde"):
        df = df[df.index >= pd.Timestamp(e["desde"], tz="UTC")]
    if e.get("hasta"):
        df = df[df.index <= pd.Timestamp(e["hasta"], tz="UTC")]
    if len(df) < 300:
        raise ValueError(f"{activo}: solo hay {len(df)} velas en el periodo elegido; hacen falta al menos 300.")
    return df


def preparar_contexto(activo: str, cfg: dict, df: pd.DataFrame | None = None) -> ContextoActivo:
    """Carga datos y prepara arrays, costes y división train/test."""
    df = cargar_datos(activo, cfg) if df is None else df
    bt = cfg["backtest"]
    costes = Costes.desde_config(bt)
    mercado = Mercado(df, cfg["experimento"]["temporalidad"], bt.get("precio_ejecucion", "apertura"))
    return ContextoActivo(
        activo=activo,
        mercado=mercado,
        corte=train_test.indice_corte(len(df), cfg["experimento"]["proporcion_train"]),
        slip=costes.array_slippage(df),
        costes=costes,
        capital=float(bt["capital_inicial"]),
        cache=CacheIndicadores(df),
    )


def calcular_posiciones(familia: str, parametros: dict, ctx: ContextoActivo, cfg: dict) -> np.ndarray:
    """Posiciones objetivo de una estrategia (con cortos y sizing según config)."""
    bt = cfg["backtest"]
    pos = crear(familia, parametros).posiciones(ctx.cache, cortos=bool(bt.get("permitir_cortos")))
    if bt.get("sizing") == "volatilidad":
        pos = aplicar_sizing_volatilidad(
            pos, ctx.mercado.close, ctx.mercado.velas_ano, float(bt["volatilidad_objetivo"])
        )
    return pos


def metricas_tramo(ctx: ContextoActivo, pos: np.ndarray, tramo: str, costes: Costes | None = None) -> np.ndarray:
    """Métricas rápidas en 'train', 'test' o 'completo'."""
    ini, fin = {"train": (0, ctx.corte), "test": (ctx.corte, ctx.n), "completo": (0, ctx.n)}[tramo]
    c = costes or ctx.costes
    slip = ctx.slip if costes is None else c.array_slippage(ctx.mercado.df)
    return backtest_rapido(ctx.mercado, pos, slip, c.comision, ctx.capital, ini, fin)


def semilla_activo(semilla: int, activo: str) -> int:
    """Semilla estable por activo (no depende del hash aleatorio de Python)."""
    return (semilla * 1_000_003 + zlib.crc32(activo.encode())) % (2**32)


# ============================================================ trabajo en paralelo
def _evaluar_lote(
    activo: str, cfg: dict, lote: list[tuple[int, str, dict]]
) -> tuple[str, list[int], np.ndarray, np.ndarray]:
    """Backtest train/test de un lote de combinaciones en un activo (se ejecuta en otro proceso)."""
    ctx = preparar_contexto(activo, cfg)
    idx = [i for i, _, _ in lote]
    train = np.zeros((len(lote), len(CLAVES_METRICAS)))
    test = np.zeros_like(train)
    for k, (_, familia, params) in enumerate(lote):
        pos = calcular_posiciones(familia, params, ctx, cfg)
        train[k] = metricas_tramo(ctx, pos, "train")
        test[k] = metricas_tramo(ctx, pos, "test")
    return activo, idx, train, test


def _referencias(ctx: ContextoActivo, cfg: dict, ops_exp: np.ndarray) -> dict:
    """Comprar y mantener + estrategias aleatorias con la misma frecuencia de operaciones."""
    unos = np.ones(ctx.n)
    salida = {
        "bh_train": metricas_tramo(ctx, unos, "train"),
        "bh_test": metricas_tramo(ctx, unos, "test"),
    }
    n_al = int(cfg["experimento"]["n_aleatorias"])
    rng = np.random.default_rng(semilla_activo(cfg["general"]["semilla"], ctx.activo))
    validas = ops_exp[ops_exp[:, 0] >= 1]
    if len(validas) == 0:
        validas = np.array([[20.0, 0.5]])
    al_train = np.zeros((n_al, len(CLAVES_METRICAS)))
    al_test = np.zeros_like(al_train)
    for j in range(n_al):
        n_ops, expo = validas[rng.integers(len(validas))]
        est = Aleatoria(
            n_operaciones=n_ops * ctx.n / max(ctx.corte, 1),
            exposicion=expo,
            semilla=int(rng.integers(2**31)),
        )
        pos = est.posiciones(ctx.cache)
        al_train[j] = metricas_tramo(ctx, pos, "train")
        al_test[j] = metricas_tramo(ctx, pos, "test")
    salida["al_train"], salida["al_test"] = al_train, al_test
    return salida


# ============================================================ datos globales
@dataclass
class Globales:
    """Información de todo el experimento que necesita la validación de cada finalista."""

    n_pruebas: int
    var_sr: float  # varianza del Sharpe por vela (train) de todas las estrategias
    descubrimientos_bh: set[int]
    aleatorias: dict[str, dict[str, np.ndarray]]  # activo → {'train': sharpe[], 'test': sharpe[]}
    buy_hold: dict[str, dict[str, dict[str, float]]]  # activo → tramo → métricas
    combinaciones_familia: dict[str, list[dict]]
    reality_check: dict = field(default_factory=dict)


def _dict_referencia(m: np.ndarray) -> dict:
    return {
        "sharpe": m[:, I["sharpe"]].round(4).tolist(),
        "rentabilidad": m[:, I["rentabilidad"]].round(4).tolist(),
        "max_dd": m[:, I["max_dd"]].round(4).tolist(),
        "n_ops": m[:, I["n_ops"]].tolist(),
        "p95_sharpe": float(np.percentile(m[:, I["sharpe"]], 95)),
        "mediana_sharpe": float(np.median(m[:, I["sharpe"]])),
        "max_sharpe": float(m[:, I["sharpe"]].max()),
    }


# ============================================================ validación completa
def validar_finalista(
    familia: str,
    parametros: dict,
    ctx: ContextoActivo,
    cfg: dict,
    glob: Globales,
    id_estrategia: int,
) -> tuple[dict, dict, dict, dict]:
    """Aplica todas las pruebas anti-humo a una estrategia.

    Devuelve (validaciones, resultados_extra, brutas, operaciones).
    """
    v_cfg, val_cfg = cfg["veredicto"], cfg["validacion"]
    # Semilla estable: depende de la estrategia y del activo, no del id de la base de datos
    semilla = semilla_activo(int(cfg["general"]["semilla"]), ctx.activo + hash_estrategia(familia, parametros))
    pos = calcular_posiciones(familia, parametros, ctx, cfg)
    res_train = backtest(ctx.mercado, pos, ctx.costes, ctx.capital, 0, ctx.corte, ctx.slip)
    res_test = backtest(ctx.mercado, pos, ctx.costes, ctx.capital, ctx.corte, ctx.n, ctx.slip)
    res_total = backtest(ctx.mercado, pos, ctx.costes, ctx.capital, 0, ctx.n, ctx.slip)
    tr, te = res_train.metricas, res_test.metricas

    # Rentabilidad bruta (sin costes) para ver cuánto se comen las comisiones
    sin = ctx.costes.sin_costes()
    brutas = {
        "train": float(metricas_tramo(ctx, pos, "train", sin)[I["rentabilidad"]]),
        "test": float(metricas_tramo(ctx, pos, "test", sin)[I["rentabilidad"]]),
    }

    val: dict[str, dict] = {}
    val["min_operaciones"] = {
        "train": int(tr["n_ops"]),
        "test": int(te["n_ops"]),
        "supera": bool(tr["n_ops"] >= v_cfg["min_operaciones"] and te["n_ops"] >= v_cfg["min_operaciones_test"]),
    }
    val["train_test"] = train_test.evaluar(tr, te, v_cfg["max_degradacion_sharpe"])

    al_test = glob.aleatorias[ctx.activo]["test"]
    umbral = float(np.percentile(al_test, v_cfg["percentil_aleatorias"]))
    val["aleatoria"] = {
        "sharpe_test": te["sharpe"],
        "umbral_test": umbral,
        "percentil_en_aleatorias": float((al_test < te["sharpe"]).mean() * 100),
        "supera": bool(te["sharpe"] > umbral),
    }

    bh_te = glob.buy_hold[ctx.activo]["test"]
    val["buy_hold"] = {
        "sharpe": te["sharpe"],
        "sharpe_bh": bh_te["sharpe"],
        "calmar": te["calmar"],
        "calmar_bh": bh_te["calmar"],
        "rentabilidad": te["rentabilidad"],
        "rentabilidad_bh": bh_te["rentabilidad"],
        "supera": bool(te["sharpe"] > bh_te["sharpe"] or te["calmar"] > bh_te["calmar"]),
    }

    mult = float(val_cfg["multiplicador_costes"])
    x2 = metricas_a_dict(metricas_tramo(ctx, pos, "test", ctx.costes.escalar(mult)))
    val["doble_coste"] = {
        "multiplicador": mult,
        "sharpe": x2["sharpe"],
        "rentabilidad": x2["rentabilidad"],
        "supera": bool(x2["sharpe"] > 0 and x2["rentabilidad"] > 0),
    }

    # Robustez: vecinos evaluados en ENTRENAMIENTO (donde se eligió la estrategia)
    clase = FAMILIAS[familia]
    espacio = _espacio_familia(familia, parametros, cfg)

    def sharpe_train(p: dict) -> float | None:
        if familia != "combinada" and not clase.valida(p):
            return None
        if familia == "combinada" and not FAMILIAS[p["base"]].valida(
            {k[2:]: v for k, v in p.items() if k.startswith("b_")}
        ):
            return None
        return float(metricas_tramo(ctx, calcular_posiciones(familia, p, ctx, cfg), "train")[I["sharpe"]])

    val["robustez"] = monte_carlo.robustez_parametros(
        parametros,
        espacio,
        sharpe_train,
        tr["sharpe"],
        val_cfg["robustez_min_positivas"],
        val_cfg["robustez_min_ratio"],
        semilla,
    )

    # Walk-forward re-optimizando la familia
    rng = np.random.default_rng(semilla)
    candidatos = walk_forward.candidatos_familia(
        glob.combinaciones_familia.get(familia, [parametros]),
        parametros,
        int(val_cfg["walk_forward_candidatos"]),
        rng,
    )
    posiciones_cand = [calcular_posiciones(familia, p, ctx, cfg) for p in candidatos]
    k = int(val_cfg["walk_forward_ventanas"])
    min_ops = max(3, round(v_cfg["min_operaciones"] * (ctx.n / (k + 1)) / max(ctx.corte, 1)))
    val["walk_forward"] = walk_forward.walk_forward(
        ctx.mercado,
        posiciones_cand,
        candidatos,
        0,
        ctx.costes.comision,
        ctx.slip,
        ctx.capital,
        k,
        min_ops,
        float(val_cfg["walk_forward_min_rentables"]),
        np.ones(ctx.n),
    )

    # Monte Carlo sobre las operaciones FUERA de muestra
    mc = monte_carlo.bootstrap_operaciones(
        res_test.operaciones["resultado_pct"].to_numpy(),
        int(val_cfg["monte_carlo_simulaciones"]),
        semilla,
        ctx.capital,
    )
    if mc.get("valido"):
        mc["supera"] = bool(mc["prob_perdida"] <= float(val_cfg["monte_carlo_max_prob_perdida"]))
    else:
        mc["supera"] = False
    val["monte_carlo"] = mc

    # Múltiples pruebas: Deflated Sharpe Ratio + Benjamini-Hochberg + reality check
    raiz = math.sqrt(ctx.mercado.velas_ano)
    r_train = res_train.rentabilidades.to_numpy()
    asim, curt = multiples_pruebas.momentos(r_train)
    sr_barra = tr["sharpe"] / raiz
    dsr, referencia = multiples_pruebas.deflated_sharpe(sr_barra, ctx.corte, asim, curt, glob.n_pruebas, glob.var_sr)
    en_bh = id_estrategia in glob.descubrimientos_bh
    explicacion = multiples_pruebas.explicar_dsr(dsr, tr["sharpe"], referencia * raiz, glob.n_pruebas)
    explicacion += " Con Benjamini-Hochberg (controlando falsos descubrimientos) " + (
        "sigue siendo significativa." if en_bh else "NO es significativa."
    )
    val["multiples_pruebas"] = {
        "dsr": dsr,
        "sharpe_referencia_anual": referencia * raiz,
        "n_pruebas": glob.n_pruebas,
        "asimetria": asim,
        "curtosis": curt,
        "benjamini_hochberg": en_bh,
        "reality_check": glob.reality_check,
        "explicacion": explicacion,
        "supera": bool(dsr >= float(val_cfg["dsr_minimo"]) and en_bh),
    }

    # Regímenes de mercado (toda la historia)
    velas_dia = ctx.mercado.velas_ano / 365.0
    reg = regimenes.clasificar(
        ctx.mercado.df["close"],
        max(2, int(val_cfg["regimen_ventana_dias"] * velas_dia)),
        float(val_cfg["regimen_umbral"]),
    )
    rent_bh = backtest(ctx.mercado, np.ones(ctx.n), ctx.costes, ctx.capital, 0, ctx.n, ctx.slip).rentabilidades
    val["regimenes"] = regimenes.por_regimen(res_total.rentabilidades, rent_bh, reg, ctx.mercado.velas_ano)

    extra = {"test_x2": x2, "completo": res_total.metricas}
    ops = {"train": res_train.operaciones, "test": res_test.operaciones}
    return val, extra, brutas, ops


def _espacio_familia(familia: str, parametros: dict, cfg: dict) -> dict:
    """Espacio de parámetros de la familia (para combinadas: el de su base con prefijo + filtro)."""
    if familia == "combinada":
        base = parametros["base"]
        espacio = {f"b_{k}": v for k, v in cfg["parametros"][base].items()}
        espacio["filtro"] = cfg["parametros"]["combinada"]["filtro"]
        return espacio
    return cfg["parametros"][familia]


# ============================================================ experimento
def ejecutar_experimento(
    cambios: dict | None = None,
    nombre: str | None = None,
    progreso: Progreso | None = None,
    lanzar_agentes: bool | None = None,
) -> int:
    """Ejecuta un experimento completo y devuelve su id."""
    t0 = time.time()
    cfg = config_experimento(cambios)
    e = cfg["experimento"]
    semilla = int(cfg["general"]["semilla"])

    def avisar(fraccion: float, texto: str) -> None:
        log.info("[%3.0f %%] %s", fraccion * 100, texto)
        if progreso:
            progreso(min(max(fraccion, 0.0), 1.0), texto)

    avisar(0.0, "Cargando datos")
    contextos = {a: preparar_contexto(a, cfg) for a in e["activos"]}
    gen = generar(e["familias"], cfg["parametros"], int(e["max_estrategias"]), semilla)
    activos = list(contextos)
    n_c = gen.n
    n_total = n_c * len(activos)
    nombre = nombre or f"{e['temporalidad']} · {len(activos)} activos · {miles(n_total)} estrategias"
    exp_id = db.crear_experimento(nombre, cfg, semilla)
    log.info("Experimento %s: %s combinaciones x %s activos", exp_id, n_c, len(activos))

    try:
        # ---------------------------------------------------------- 1) backtests masivos
        train = {a: np.zeros((n_c, len(CLAVES_METRICAS))) for a in activos}
        test = {a: np.zeros((n_c, len(CLAVES_METRICAS))) for a in activos}
        nucleos = int(cfg["general"].get("nucleos", -1))
        if n_total < 400:
            nucleos = 1
        n_trab = nucleos if nucleos > 0 else max(1, (__import__("os").cpu_count() or 1))
        tam_lote = max(25, math.ceil(n_c / max(1, 4 * n_trab // max(1, len(activos)))))
        tareas = [
            delayed(_evaluar_lote)(a, cfg, [(i, *gen.combinaciones[i]) for i in range(ini, min(ini + tam_lote, n_c))])
            for a in activos
            for ini in range(0, n_c, tam_lote)
        ]
        hechos = 0
        t_bt = time.time()
        for activo, idx, mtr, mte in Parallel(n_jobs=nucleos, return_as="generator_unordered")(tareas):
            train[activo][idx] = mtr
            test[activo][idx] = mte
            hechos += len(idx)
            transcurrido = time.time() - t_bt
            resta = transcurrido / hechos * (n_total - hechos)
            avisar(
                0.02 + 0.68 * hechos / n_total,
                f"Backtests: {miles(hechos)}/{miles(n_total)} · quedan ~{resta:.0f} s",
            )

        # ---------------------------------------------------------- 2) referencias de control
        avisar(0.71, "Estrategias de control: comprar y mantener y aleatorias")
        refs = {}
        for a, ctx in contextos.items():
            ops_exp = train[a][:, [I["n_ops"], I["exposicion"]]]
            refs[a] = _referencias(ctx, cfg, ops_exp)
            for tramo in ("train", "test"):
                repo.guardar_referencia(exp_id, a, tramo, "buy_hold", metricas_a_dict(refs[a][f"bh_{tramo}"]))
                repo.guardar_referencia(exp_id, a, tramo, "aleatoria", _dict_referencia(refs[a][f"al_{tramo}"]))

        # ---------------------------------------------------------- 3) guardar todo
        avisar(0.75, "Guardando resultados")
        filas = [(f, p, hash_estrategia(f, p), a, e["temporalidad"]) for a in activos for f, p in gen.combinaciones]
        ids = repo.insertar_estrategias(exp_id, filas)
        id_de = {(a, i): ids[k * n_c + i] for k, a in enumerate(activos) for i in range(n_c)}
        repo.insertar_resultados(
            [(id_de[(a, i)], "train", train[a][i]) for a in activos for i in range(n_c)]
            + [(id_de[(a, i)], "test", test[a][i]) for a in activos for i in range(n_c)]
        )

        # ---------------------------------------------------------- 4) múltiples pruebas (global)
        sr_barra = np.concatenate(
            [train[a][:, I["sharpe"]] / math.sqrt(contextos[a].mercado.velas_ano) for a in activos]
        )
        n_obs = np.concatenate([np.full(n_c, contextos[a].corte) for a in activos])
        ids_orden = [id_de[(a, i)] for a in activos for i in range(n_c)]
        pvals = np.array([multiples_pruebas.pvalores_sharpe(np.array([s]), int(o))[0] for s, o in zip(sr_barra, n_obs)])
        mascara_bh = multiples_pruebas.benjamini_hochberg(pvals, float(cfg["validacion"]["alfa_bh"]))
        sr_train_todas = np.concatenate([train[a][:, I["sharpe"]] for a in activos])
        sr_al_train = np.concatenate([refs[a]["al_train"][:, I["sharpe"]] for a in activos])
        rc = multiples_pruebas.reality_check(float(sr_train_todas.max()), sr_al_train, n_total, 1000, semilla)
        rc["mejor_sharpe_real"] = float(sr_train_todas.max())
        glob = Globales(
            n_pruebas=n_total,
            var_sr=float(np.var(sr_barra[np.isfinite(sr_barra)])),
            descubrimientos_bh={i for i, m in zip(ids_orden, mascara_bh) if m},
            aleatorias={
                a: {"train": refs[a]["al_train"][:, I["sharpe"]], "test": refs[a]["al_test"][:, I["sharpe"]]}
                for a in activos
            },
            buy_hold={
                a: {"train": metricas_a_dict(refs[a]["bh_train"]), "test": metricas_a_dict(refs[a]["bh_test"])}
                for a in activos
            },
            combinaciones_familia=_por_familia(gen.combinaciones),
            reality_check=rc,
        )

        # ---------------------------------------------------------- 5) veredicto básico + finalistas
        avisar(0.78, "Veredictos y selección de finalistas (solo con entrenamiento)")
        pct_al = cfg["veredicto"]["percentil_aleatorias"]
        umbral_train = {a: float(np.percentile(glob.aleatorias[a]["train"], pct_al)) for a in activos}
        veredictos: dict[int, tuple[str, list[str]]] = {}
        candidatas: list[tuple[float, str, int]] = []
        for a in activos:
            for i in range(n_c):
                mtr = dict(zip(CLAVES_METRICAS, train[a][i]))
                mte = dict(zip(CLAVES_METRICAS, test[a][i]))
                veredictos[id_de[(a, i)]] = veredicto.veredicto_basico(mtr, mte, umbral_train[a], cfg)
                if mtr["n_ops"] >= cfg["veredicto"]["min_operaciones"]:
                    candidatas.append((mtr["sharpe"], a, i))
        candidatas.sort(key=lambda x: (-x[0], x[1], x[2]))
        top = candidatas[: int(e["top_finalistas"])]
        repo.marcar_finalistas([id_de[(a, i)] for _, a, i in top])

        # ---------------------------------------------------------- 6) validación completa
        for k, (_, a, i) in enumerate(top):
            avisar(0.80 + 0.17 * k / max(len(top), 1), f"Validando finalista {k + 1}/{len(top)}")
            est_id = id_de[(a, i)]
            familia, params = gen.combinaciones[i]
            val, extra, brutas, ops = validar_finalista(familia, params, contextos[a], cfg, glob, est_id)
            guardar_validacion_completa(est_id, val, extra, brutas, ops, train[a][i], test[a][i])
            veredictos[est_id] = veredicto.veredicto_completo(val, cfg)

        repo.guardar_veredictos([(i, v, v, m) for i, (v, m) in veredictos.items()])

        # ---------------------------------------------------------- 7) resumen
        conteo = {v: 0 for v in (veredicto.APROBADA, veredicto.SOSPECHOSA, veredicto.DESCARTADA)}
        for v, _ in veredictos.values():
            conteo[v] += 1
        sr_test_todas = np.concatenate([test[a][:, I["sharpe"]] for a in activos])
        umbral_test = np.concatenate([np.full(n_c, np.percentile(glob.aleatorias[a]["test"], pct_al)) for a in activos])
        resumen = {
            "n_combinaciones": n_c,
            "n_activos": len(activos),
            "n_estrategias": n_total,
            "espacio_total": gen.espacio_total,
            "por_familia": gen.por_familia,
            "conteo": conteo,
            "pct_descartadas": conteo[veredicto.DESCARTADA] / n_total,
            "pct_sospechosas": conteo[veredicto.SOSPECHOSA] / n_total,
            "pct_aprobadas": conteo[veredicto.APROBADA] / n_total,
            "fuente": e["fuente"],
            "datos_sinteticos": e["fuente"] == "sintetico",
            "temporalidad": e["temporalidad"],
            "activos": {
                a: {
                    "velas": ctx.n,
                    "velas_ano": ctx.mercado.velas_ano,
                    "corte": ctx.corte,
                    "desde": str(ctx.mercado.indice[0]),
                    "inicio_test": str(ctx.mercado.indice[ctx.corte]),
                    "hasta": str(ctx.mercado.indice[-1]),
                    "bh_train": glob.buy_hold[a]["train"],
                    "bh_test": glob.buy_hold[a]["test"],
                }
                for a, ctx in contextos.items()
            },
            "aleatorias": {
                "frac_reales_sobre_p95_test": float((sr_test_todas > umbral_test).mean()),
                "frac_reales_sobre_p95_train": float(
                    np.mean(np.concatenate([train[a][:, I["sharpe"]] > umbral_train[a] for a in activos]))
                ),
                "mejor_sharpe_real_train": float(sr_train_todas.max()),
                "mejor_sharpe_aleatoria_train": float(sr_al_train.max()),
                "mediana_real_test": float(np.median(sr_test_todas)),
                "mediana_aleatoria_test": float(
                    np.median(np.concatenate([glob.aleatorias[a]["test"] for a in activos]))
                ),
                "reality_check": rc,
            },
            "multiples_pruebas": {
                "descubrimientos_bh": int(mascara_bh.sum()),
                "fraccion_bh": float(mascara_bh.mean()),
                "bonferroni": int(multiples_pruebas.bonferroni(pvals).sum()),
                "sharpe_max_esperado_azar": float(
                    multiples_pruebas.sharpe_maximo_esperado(n_total, glob.var_sr)
                    * math.sqrt(np.mean([c.mercado.velas_ano for c in contextos.values()]))
                ),
            },
            "finalistas": [id_de[(a, i)] for _, a, i in top],
        }
        duracion = time.time() - t0
        db.cerrar_experimento(exp_id, "terminado", duracion, resumen, n_c, n_total, gen.espacio_total)
        avisar(0.97, "Backtests y validación terminados")

        # ---------------------------------------------------------- 8) agentes (opcional)
        usar_agentes = cfg["agentes"]["activado"] if lanzar_agentes is None else lanzar_agentes
        if usar_agentes and top:
            if clave_anthropic():
                from modulos.trading.agentes.panel import ejecutar_panel

                try:
                    avisar(0.97, "Panel de agentes críticos")
                    info = ejecutar_panel(exp_id, cfg=cfg, progreso=lambda f, t: avisar(0.97 + 0.03 * f, t))
                    resumen["agentes"] = info
                except Exception as err:  # noqa: BLE001 — los agentes nunca rompen el experimento
                    log.exception("El panel de agentes falló")
                    resumen["agentes"] = {"error": str(err)}
            else:
                resumen["agentes"] = {"omitido": "No hay ANTHROPIC_API_KEY en .env"}
            resumen["conteo"] = repo.conteo_veredictos(exp_id)
            db.actualizar_resumen(exp_id, resumen)
        avisar(1.0, f"Terminado en {time.time() - t0:.0f} s")
        return exp_id
    except Exception as err:
        db.cerrar_experimento(exp_id, "error", time.time() - t0, {"error": str(err)}, n_c, n_total)
        raise


def _por_familia(combinaciones: list[tuple[str, dict]]) -> dict[str, list[dict]]:
    salida: dict[str, list[dict]] = {}
    for f, p in combinaciones:
        salida.setdefault(f, []).append(p)
    return salida


def guardar_validacion_completa(
    est_id: int,
    val: dict,
    extra: dict,
    brutas: dict,
    ops: dict,
    m_train: np.ndarray,
    m_test: np.ndarray,
) -> None:
    """Persiste validaciones, operaciones y métricas adicionales de una finalista."""
    repo.guardar_validaciones(est_id, val)
    repo.insertar_resultados(
        [(est_id, "train", m_train), (est_id, "test", m_test)] + [(est_id, tramo, m) for tramo, m in extra.items()],
        brutas={(est_id, t): v for t, v in brutas.items()},
    )
    for tramo, df in ops.items():
        repo.guardar_operaciones(est_id, tramo, df)


# ============================================================ validación a demanda
def contexto_de_estrategia(estrategia_id: int) -> tuple[dict, dict, ContextoActivo, dict]:
    """(info de la estrategia, config del experimento, contexto del activo, experimento)."""
    info = repo.estrategia(estrategia_id)
    if info is None:
        raise KeyError(f"No existe la estrategia {estrategia_id}")
    exp = db.obtener_experimento(int(info["experimento_id"]))
    cfg = exp["config"]
    ctx = preparar_contexto(info["activo"], cfg)
    return info, cfg, ctx, exp


def validar_a_fondo(estrategia_id: int) -> tuple[str, list[str]]:
    """Validación completa de una estrategia que no fue finalista (desde la página de detalle)."""
    info, cfg, ctx, exp = contexto_de_estrategia(estrategia_id)
    ranking = repo.ranking(int(info["experimento_id"]))
    refs = repo.referencias(int(info["experimento_id"]))
    resumen = exp["resumen"]
    raices = {a: math.sqrt(d["velas_ano"]) for a, d in resumen["activos"].items()}
    cortes = {a: d["corte"] for a, d in resumen["activos"].items()}
    sr_barra = ranking["sharpe_train"].to_numpy() / ranking["activo"].map(raices).to_numpy()
    pvals = np.array(
        [
            multiples_pruebas.pvalores_sharpe(np.array([s]), int(cortes[a]))[0]
            for s, a in zip(sr_barra, ranking["activo"])
        ]
    )
    mascara = multiples_pruebas.benjamini_hochberg(pvals, float(cfg["validacion"]["alfa_bh"]))
    aleatorias: dict[str, dict[str, np.ndarray]] = {}
    buy_hold: dict[str, dict[str, dict]] = {}
    for r in refs.itertuples():
        if r.tipo == "aleatoria":
            aleatorias.setdefault(r.activo, {})[r.tramo] = np.array(r.datos["sharpe"])
        else:
            buy_hold.setdefault(r.activo, {})[r.tramo] = r.datos
    gen = generar(
        cfg["experimento"]["familias"],
        cfg["parametros"],
        int(cfg["experimento"]["max_estrategias"]),
        int(cfg["general"]["semilla"]),
    )
    glob = Globales(
        n_pruebas=int(resumen["n_estrategias"]),
        var_sr=float(np.nanvar(sr_barra)),
        descubrimientos_bh=set(ranking["id"][mascara].tolist()),
        aleatorias=aleatorias,
        buy_hold=buy_hold,
        combinaciones_familia=_por_familia(gen.combinaciones),
        reality_check=resumen.get("aleatorias", {}).get("reality_check", {}),
    )
    pos = calcular_posiciones(info["familia"], info["parametros"], ctx, cfg)
    val, extra, brutas, ops = validar_finalista(info["familia"], info["parametros"], ctx, cfg, glob, estrategia_id)
    guardar_validacion_completa(
        estrategia_id, val, extra, brutas, ops, metricas_tramo(ctx, pos, "train"), metricas_tramo(ctx, pos, "test")
    )
    v, motivos = veredicto.veredicto_completo(val, cfg)
    repo.guardar_veredictos([(estrategia_id, v, v, motivos)])
    return v, motivos


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Lanza un experimento con la configuración de config.yaml")
    p.add_argument("--fuente", choices=["binance", "sintetico", "yfinance", "csv"])
    p.add_argument("--temporalidad", choices=["1h", "4h", "1d"])
    p.add_argument("--max", type=int, help="Número máximo de combinaciones de parámetros")
    p.add_argument("--sin-agentes", action="store_true")
    args = p.parse_args()
    cambios: dict[str, Any] = {"experimento": {}}
    if args.fuente:
        cambios["experimento"]["fuente"] = args.fuente
    if args.temporalidad:
        cambios["experimento"]["temporalidad"] = args.temporalidad
    if args.max:
        cambios["experimento"]["max_estrategias"] = args.max
    ident = ejecutar_experimento(cambios, lanzar_agentes=False if args.sin_agentes else None)
    exp = db.obtener_experimento(ident)
    r = exp["resumen"]
    print(f"\nExperimento {ident} terminado en {exp['duracion_s']:.0f} s")
    print(f"Estrategias probadas: {miles(r['n_estrategias'])}")
    for v, n in r["conteo"].items():
        print(f"  {v:<11} {miles(n):>7}  ({decimal(n / r['n_estrategias'] * 100)} %)")
    if r.get("datos_sinteticos"):
        print("\nAVISO: son datos SINTÉTICOS. Ningún resultado dice nada del mercado real.")
