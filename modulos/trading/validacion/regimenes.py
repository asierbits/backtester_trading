"""Análisis por regímenes de mercado: alcista, bajista y lateral.

Cada vela se etiqueta según la rentabilidad del activo en la ventana anterior
(p. ej. 90 días): > +10 % alcista, < −10 % bajista, lo demás lateral. Así se ve si
una estrategia solo ganó porque el mercado subía.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ETIQUETAS = {1: "alcista", -1: "bajista", 0: "lateral"}


def clasificar(close: pd.Series, ventana_velas: int, umbral: float) -> pd.Series:
    """Régimen de cada vela (1, -1, 0) según la rentabilidad de la ventana anterior."""
    rent = close / close.shift(ventana_velas) - 1
    regimen = pd.Series(0, index=close.index)
    regimen[rent > umbral] = 1
    regimen[rent < -umbral] = -1
    return regimen


def por_regimen(rent_estrategia: pd.Series, rent_bh: pd.Series, regimen: pd.Series, velas_ano: float) -> dict:
    """Rentabilidad anualizada (media simple × velas/año) de estrategia y B&H en cada régimen."""
    salida: dict = {"regimenes": {}}
    regimen = regimen.reindex(rent_estrategia.index).fillna(0).astype(int)
    for codigo, nombre in ETIQUETAS.items():
        mascara = regimen == codigo
        n = int(mascara.sum())
        if n == 0:
            salida["regimenes"][nombre] = {"velas": 0}
            continue
        e = rent_estrategia[mascara]
        b = rent_bh.reindex(e.index)
        salida["regimenes"][nombre] = {
            "velas": n,
            "pct_tiempo": n / len(regimen),
            "rent_anual_estrategia": float(e.mean() * velas_ano),
            "rent_anual_bh": float(b.mean() * velas_ano),
            "rent_acumulada_estrategia": float(np.prod(1 + e) - 1),
            "rent_acumulada_bh": float(np.prod(1 + b) - 1),
        }
    r = salida["regimenes"]
    # Supera si en los mercados no alcistas no pierde dinero o lo hace mejor que comprar y mantener
    fallos = []
    for nombre in ("bajista", "lateral"):
        d = r.get(nombre, {})
        if d.get("velas", 0) >= 30:
            if d["rent_anual_estrategia"] < 0 and d["rent_anual_estrategia"] < d["rent_anual_bh"]:
                fallos.append(nombre)
    alc = r.get("alcista", {})
    solo_alcista = bool(
        alc.get("velas", 0) > 0
        and alc.get("rent_anual_estrategia", 0) > 0
        and all(r.get(n, {}).get("rent_anual_estrategia", 0) <= 0 for n in ("bajista", "lateral"))
    )
    salida["fallos"] = fallos
    salida["solo_gana_en_alcista"] = solo_alcista
    salida["supera"] = not fallos and not solo_alcista
    return salida
