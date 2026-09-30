"""Monte Carlo y robustez.

1. **Bootstrap de operaciones**: se remuestrean con reemplazo las operaciones miles de
   veces para obtener intervalos de confianza de la rentabilidad y de la caída máxima.
   Responde a: ¿y si el orden o la suerte de las operaciones hubiera sido otro?
2. **Perturbación de parámetros**: una estrategia sólida no se hunde si cambias la
   media de 20 a 22. Se prueban vecinos de los parámetros y se mide cuántos aguantan.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from modulos.trading.estrategias.base import expandir


def bootstrap_operaciones(rent_ops: np.ndarray, n_sim: int, semilla: int, capital: float = 1000.0) -> dict:
    """Intervalos de confianza por remuestreo de operaciones (rentabilidades en fracción)."""
    r = np.asarray(rent_ops, dtype=float)
    if len(r) < 2:
        return {"valido": False, "motivo": "Menos de 2 operaciones: no se puede remuestrear"}
    rng = np.random.default_rng(semilla)
    muestras = rng.choice(r, size=(n_sim, len(r)), replace=True)
    trayectorias = capital * np.cumprod(1 + muestras, axis=1)
    finales = trayectorias[:, -1] / capital - 1
    picos = np.maximum.accumulate(np.concatenate([np.full((n_sim, 1), capital), trayectorias], axis=1), axis=1)[:, 1:]
    dds = (1 - trayectorias / picos).max(axis=1)
    # Abanico: percentiles por número de operación
    percentiles = {str(p): [float(x) for x in np.percentile(trayectorias, p, axis=0)] for p in (5, 25, 50, 75, 95)}
    muestra_tray = trayectorias[: min(50, n_sim)].round(2).tolist()
    return {
        "valido": True,
        "n_sim": n_sim,
        "n_operaciones": int(len(r)),
        "rentabilidad_p5": float(np.percentile(finales, 5)),
        "rentabilidad_p50": float(np.percentile(finales, 50)),
        "rentabilidad_p95": float(np.percentile(finales, 95)),
        "max_dd_p50": float(np.percentile(dds, 50)),
        "max_dd_p95": float(np.percentile(dds, 95)),
        "prob_perdida": float((finales < 0).mean()),
        "abanico": percentiles,
        "trayectorias": muestra_tray,
    }


def vecinos(
    parametros: dict[str, Any], espacio: dict[str, Any], n_aleatorios: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Parámetros perturbados: ±δ y ±2δ en cada parámetro numérico (δ ≈ 10 % o un paso),
    más perturbaciones conjuntas al azar."""
    numericos: dict[str, tuple[float, float, float, bool]] = {}
    for clave, spec in espacio.items():
        valores = expandir(spec)
        if clave in parametros and len(valores) > 1 and all(isinstance(v, (int, float)) for v in valores):
            paso = float(spec[2]) if isinstance(spec, list) and len(spec) == 3 else float(valores[1] - valores[0])
            es_entero = all(isinstance(v, int) for v in valores)
            numericos[clave] = (float(min(valores)), float(max(valores)), paso, es_entero)

    def delta(clave: str) -> float:
        _, _, paso, _ = numericos[clave]
        v = abs(float(parametros[clave]))
        return max(paso, round(0.1 * v / paso) * paso)

    def ajustar(clave: str, valor: float) -> float | int:
        minimo, maximo, _, es_entero = numericos[clave]
        # Se permite salir un poco del rango de búsqueda (la robustez no entiende de rejillas)
        valor = max(valor, minimo * 0.5 if minimo > 0 else minimo)
        return int(round(valor)) if es_entero else round(float(valor), 6)

    salida: list[dict[str, Any]] = []
    for clave in numericos:
        for mult in (-2, -1, 1, 2):
            p = dict(parametros)
            p[clave] = ajustar(clave, parametros[clave] + mult * delta(clave))
            if p != parametros:
                salida.append(p)
    claves = list(numericos)
    for _ in range(n_aleatorios if claves else 0):
        p = dict(parametros)
        for clave in claves:
            p[clave] = ajustar(clave, parametros[clave] + rng.integers(-1, 2) * delta(clave))
        if p != parametros:
            salida.append(p)
    # Sin duplicados, en orden estable
    unicos, vistos = [], set()
    for p in salida:
        clave = tuple(sorted(p.items()))
        if clave not in vistos:
            vistos.add(clave)
            unicos.append(p)
    return unicos


def robustez_parametros(
    parametros: dict[str, Any],
    espacio: dict[str, Any],
    evaluar: Callable[[dict[str, Any]], float | None],
    sharpe_original: float,
    min_positivas: float,
    min_ratio: float,
    semilla: int,
) -> dict:
    """Evalúa los vecinos con `evaluar` (Sharpe en entrenamiento) y resume la robustez."""
    rng = np.random.default_rng(semilla)
    lista = vecinos(parametros, espacio, 10, rng)
    resultados = []
    for p in lista:
        sr = evaluar(p)
        if sr is not None:
            resultados.append({"parametros": p, "sharpe": float(sr)})
    if not resultados:
        return {"valido": False, "supera": True, "motivo": "Sin parámetros numéricos que perturbar"}
    srs = np.array([r["sharpe"] for r in resultados])
    frac_pos = float((srs > 0).mean())
    ratio = float(np.median(srs) / sharpe_original) if sharpe_original > 1e-9 else 0.0
    return {
        "valido": True,
        "n_vecinos": len(resultados),
        "frac_positivas": frac_pos,
        "ratio_mediana": ratio,
        "sharpe_original": float(sharpe_original),
        "sharpe_min": float(srs.min()),
        "sharpe_mediana": float(np.median(srs)),
        "vecinos": resultados,
        "supera": bool(frac_pos >= min_positivas and ratio >= min_ratio),
    }
