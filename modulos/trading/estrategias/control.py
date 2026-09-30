"""Estrategias de control: comprar y mantener, y estrategia aleatoria.

No se optimizan: sirven de vara de medir. Si una estrategia no supera claramente a
la aleatoria, su resultado es compatible con la pura suerte.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from modulos.trading.estrategias.base import Estrategia
from modulos.trading.estrategias.indicadores import CacheIndicadores


class CompraMantiene(Estrategia):
    familia = "buy_hold"
    nombre_legible = "Comprar y mantener"
    descripcion = "Compra en la primera vela y no vende nunca."

    def _posiciones(self, c: CacheIndicadores, cortos: bool) -> np.ndarray:
        return np.ones_like(c.close)


@njit(cache=True)
def _cadena_markov(n: int, p_entrada: float, p_salida: float, azar: np.ndarray) -> np.ndarray:
    pos = np.zeros(n, dtype=np.float64)
    actual = 0.0
    for i in range(n):
        if actual == 0.0:
            if azar[i] < p_entrada:
                actual = 1.0
        elif azar[i] < p_salida:
            actual = 0.0
        pos[i] = actual
    return pos


class Aleatoria(Estrategia):
    """Entra y sale al azar con una frecuencia de operaciones y exposición objetivo.

    Con n velas, k operaciones y exposición e, la duración media de una operación es
    e·n/k y la de los periodos fuera es (1−e)·n/k, de donde salen las probabilidades
    de salida y de entrada de una cadena de Markov de dos estados.
    """

    familia = "aleatoria"
    nombre_legible = "Aleatoria (control)"
    descripcion = "Entra y sale al azar con la misma frecuencia de operaciones que las reales."

    def _posiciones(self, c: CacheIndicadores, cortos: bool) -> np.ndarray:
        n = len(c.close)
        k = max(float(self.parametros["n_operaciones"]), 1.0)
        e = float(np.clip(self.parametros["exposicion"], 0.02, 0.98))
        p_salida = min(1.0, k / (e * n))
        p_entrada = min(1.0, k / ((1 - e) * n))
        rng = np.random.default_rng(int(self.parametros["semilla"]))
        return _cadena_markov(n, p_entrada, p_salida, rng.random(n))
