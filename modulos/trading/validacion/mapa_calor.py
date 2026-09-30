"""Mapa de calor de parámetros: rendimiento en función de dos parámetros.

Las mesetas (zonas amplias con buen resultado) indican una ventaja más creíble; un
pico aislado rodeado de malos resultados es la firma típica del sobreajuste.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from modulos.trading.estrategias.base import expandir


def reducir(valores: list, maximo: int) -> list:
    """Submuestrea una lista de valores de forma uniforme."""
    if len(valores) <= maximo:
        return valores
    idx = np.unique(np.linspace(0, len(valores) - 1, maximo).round().astype(int))
    return [valores[i] for i in idx]


def mapa_calor(
    parametros: dict[str, Any],
    espacio: dict[str, Any],
    param_x: str,
    param_y: str,
    evaluar: Callable[[dict[str, Any]], float | None],
    maximo_por_eje: int = 15,
) -> pd.DataFrame:
    """Matriz (filas = param_y, columnas = param_x) con la métrica devuelta por `evaluar`."""
    # Los valores de la propia estrategia siempre forman parte de los ejes
    xs = sorted(set(reducir(expandir(espacio[param_x]), maximo_por_eje)) | {parametros[param_x]})
    ys = sorted(set(reducir(expandir(espacio[param_y]), maximo_por_eje)) | {parametros[param_y]})
    tabla = pd.DataFrame(np.nan, index=ys, columns=xs, dtype=float)
    for y in ys:
        for x in xs:
            p = dict(parametros)
            p[param_x], p[param_y] = x, y
            valor = evaluar(p)
            if valor is not None:
                tabla.loc[y, x] = valor
    tabla.index.name, tabla.columns.name = param_y, param_x
    return tabla


def estabilidad(tabla: pd.DataFrame, valor_x: Any, valor_y: Any) -> float | None:
    """Media de los vecinos inmediatos del punto elegido respecto a su valor (1 = meseta)."""
    if valor_y not in tabla.index or valor_x not in tabla.columns:
        return None
    i, j = tabla.index.get_loc(valor_y), tabla.columns.get_loc(valor_x)
    centro = tabla.iloc[i, j]
    if not np.isfinite(centro) or centro <= 0:
        return None
    zona = tabla.iloc[max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2].to_numpy().ravel()
    zona = zona[np.isfinite(zona)]
    return float((zona.sum() - centro) / max(len(zona) - 1, 1) / centro)
