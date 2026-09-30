"""Generador de variaciones de parámetros.

Crea hasta `max_estrategias` combinaciones repartidas entre las familias elegidas:
- Si el espacio completo cabe, se prueba entero (búsqueda en rejilla).
- Si no, se toma una muestra aleatoria sin repetición (reproducible con la semilla).

Siempre devuelve también el **tamaño del espacio** y el **número de combinaciones
probadas**, que hacen falta para corregir por múltiples pruebas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modulos.trading.estrategias.base import FAMILIAS
from modulos.trading.estrategias.combinada import Combinada


@dataclass
class Generacion:
    """Resultado del generador."""

    combinaciones: list[tuple[str, dict[str, Any]]]
    espacio_total: int
    por_familia: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.combinaciones)


def rejilla_familia(familia: str, parametros_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Todas las combinaciones válidas de una familia según config.yaml."""
    if familia == "combinada":
        return Combinada.rejilla_con_bases(parametros_cfg["combinada"], parametros_cfg)
    return FAMILIAS[familia].rejilla(parametros_cfg[familia])


def repartir_cuotas(tamanos: dict[str, int], maximo: int) -> dict[str, int]:
    """Reparte 'maximo' entre familias; las pequeñas se prueban enteras y sobran huecos para el resto."""
    cuotas = {f: 0 for f in tamanos}
    pendientes = {f for f, t in tamanos.items() if t > 0}
    restante = maximo
    while pendientes and restante > 0:
        parte = max(restante // len(pendientes), 1)
        for f in sorted(pendientes):
            extra = min(parte, tamanos[f] - cuotas[f], restante)
            cuotas[f] += extra
            restante -= extra
        pendientes = {f for f in pendientes if cuotas[f] < tamanos[f]}
    return cuotas


def generar(
    familias: list[str],
    parametros_cfg: dict[str, Any],
    max_estrategias: int,
    semilla: int,
) -> Generacion:
    """Genera las combinaciones (familia, parámetros) a probar."""
    rng = np.random.default_rng(semilla)
    rejillas = {f: rejilla_familia(f, parametros_cfg) for f in familias}
    tamanos = {f: len(r) for f, r in rejillas.items()}
    cuotas = repartir_cuotas(tamanos, max_estrategias)

    combinaciones: list[tuple[str, dict[str, Any]]] = []
    por_familia: dict[str, dict[str, int]] = {}
    for f in familias:
        rejilla, cuota = rejillas[f], cuotas[f]
        if cuota >= len(rejilla):
            elegidas = rejilla
        else:
            indices = np.sort(rng.choice(len(rejilla), size=cuota, replace=False))
            elegidas = [rejilla[i] for i in indices]
        combinaciones.extend((f, p) for p in elegidas)
        por_familia[f] = {"espacio": len(rejilla), "probadas": len(elegidas)}
    return Generacion(combinaciones, sum(tamanos.values()), por_familia)
