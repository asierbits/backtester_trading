"""División cronológica entrenamiento / test y medida de la degradación.

El tramo de test NUNCA se usa para elegir estrategias: la selección se hace solo con
el entrenamiento y el test mide cuánto se parece el futuro a lo que prometía el pasado.
"""

from __future__ import annotations

import numpy as np


def indice_corte(n: int, proporcion_train: float) -> int:
    """Índice de la primera vela de test."""
    if not 0.1 <= proporcion_train <= 0.9:
        raise ValueError("La proporción de entrenamiento debe estar entre 0,1 y 0,9")
    return int(n * proporcion_train)


def conservacion_sharpe(sharpe_train: float, sharpe_test: float) -> float:
    """Fracción del Sharpe de entrenamiento que se conserva en test (1 = igual, <0 = se invierte)."""
    if sharpe_train <= 1e-9:
        return np.nan
    return float(sharpe_test / sharpe_train)


def evaluar(train: dict, test: dict, max_degradacion: float) -> dict:
    """Resultado de la prueba train/test para una estrategia."""
    conserva = conservacion_sharpe(train["sharpe"], test["sharpe"])
    degradacion = 1 - conserva if not np.isnan(conserva) else np.nan
    supera = bool(test["sharpe"] > 0 and not np.isnan(conserva) and conserva >= max_degradacion)
    return {
        "sharpe_train": train["sharpe"],
        "sharpe_test": test["sharpe"],
        "rentabilidad_train": train["rentabilidad"],
        "rentabilidad_test": test["rentabilidad"],
        "conservacion": conserva,
        "degradacion": degradacion,
        "supera": supera,
    }
