"""Corrección por múltiples pruebas: si pruebas miles de estrategias, alguna parecerá
buena por pura suerte. Aquí se mide cuánto.

- **Sharpe probabilístico (PSR)**: probabilidad de que el Sharpe real sea > 0 dado el
  Sharpe observado, el nº de observaciones, la asimetría y la curtosis (Bailey y López de Prado).
- **Deflated Sharpe Ratio (DSR)**: PSR pero comparando con el Sharpe máximo que cabría
  esperar por azar tras N pruebas, en lugar de con 0.
- **Benjamini-Hochberg**: controla la proporción de falsos descubrimientos entre las
  estrategias que parecen significativas.
- **Reality check sencillo**: compara el mejor Sharpe real con la distribución del
  mejor Sharpe de estrategias aleatorias.

Todos los Sharpe de este módulo son **por vela** (sin anualizar) salvo que se indique.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_NORMAL = NormalDist()
EULER = 0.5772156649015329


def cdf(x: float) -> float:
    return _NORMAL.cdf(x)


def ppf(p: float) -> float:
    return _NORMAL.inv_cdf(min(max(p, 1e-12), 1 - 1e-12))


def momentos(rentabilidades: np.ndarray) -> tuple[float, float]:
    """Asimetría y curtosis (no en exceso: la normal tiene 3)."""
    r = np.asarray(rentabilidades, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or r.std() == 0:
        return 0.0, 3.0
    z = (r - r.mean()) / r.std()
    return float((z**3).mean()), float((z**4).mean())


def sharpe_probabilistico(
    sr: float, n_obs: int, asimetria: float = 0.0, curtosis: float = 3.0, sr_ref: float = 0.0
) -> float:
    """PSR: P(Sharpe real > sr_ref). Sharpe por vela."""
    if n_obs < 3:
        return 0.0
    denominador = 1 - asimetria * sr + (curtosis - 1) / 4 * sr**2
    denominador = math.sqrt(max(denominador, 1e-12))
    return cdf((sr - sr_ref) * math.sqrt(n_obs - 1) / denominador)


def sharpe_maximo_esperado(n_pruebas: int, varianza_sr: float) -> float:
    """Sharpe máximo esperado por azar tras n_pruebas independientes (por vela)."""
    if n_pruebas < 2 or varianza_sr <= 0:
        return 0.0
    return math.sqrt(varianza_sr) * ((1 - EULER) * ppf(1 - 1 / n_pruebas) + EULER * ppf(1 - 1 / (n_pruebas * math.e)))


def deflated_sharpe(
    sr: float,
    n_obs: int,
    asimetria: float,
    curtosis: float,
    n_pruebas: int,
    varianza_sr: float,
) -> tuple[float, float]:
    """Devuelve (DSR, Sharpe de referencia usado)."""
    referencia = sharpe_maximo_esperado(n_pruebas, varianza_sr)
    return sharpe_probabilistico(sr, n_obs, asimetria, curtosis, referencia), referencia


def pvalores_sharpe(sr: np.ndarray, n_obs: int) -> np.ndarray:
    """p-valor unilateral de H0: Sharpe ≤ 0 (aprox. normal, sin momentos superiores)."""
    sr = np.asarray(sr, dtype=float)
    z = sr * math.sqrt(max(n_obs - 1, 1)) / np.sqrt(np.maximum(1 + 0.5 * sr**2, 1e-12))
    return np.array([1 - cdf(v) for v in z])


def benjamini_hochberg(pvalores: np.ndarray, alfa: float = 0.05) -> np.ndarray:
    """Máscara de hipótesis rechazadas (descubrimientos) con control FDR."""
    p = np.asarray(pvalores, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    orden = np.argsort(p)
    umbrales = alfa * np.arange(1, m + 1) / m
    bajo = p[orden] <= umbrales
    rechazadas = np.zeros(m, dtype=bool)
    if bajo.any():
        k = np.max(np.nonzero(bajo)[0])
        rechazadas[orden[: k + 1]] = True
    return rechazadas


def bonferroni(pvalores: np.ndarray, alfa: float = 0.05) -> np.ndarray:
    """Máscara de rechazadas con Bonferroni (muy conservador)."""
    p = np.asarray(pvalores, dtype=float)
    return p <= alfa / max(len(p), 1)


def reality_check(
    mejor_sr: float,
    sr_aleatorias: np.ndarray,
    n_pruebas: int,
    n_remuestreos: int = 2000,
    semilla: int = 0,
) -> dict:
    """Compara el mejor Sharpe real con la distribución del 'mejor de N' estrategias aleatorias.

    Se ajusta una normal a los Sharpe de las estrategias aleatorias y se simula el máximo
    de N extracciones (N = estrategias probadas) muchas veces: así se obtiene la
    distribución del mejor resultado que daría la pura suerte probando tanto como tú.
    El p-valor es la fracción de veces que la suerte iguala o supera al mejor real.
    Es aproximado y conservador: las estrategias reales están correlacionadas entre sí,
    así que el número efectivo de pruebas independientes es menor que N.
    """
    pool = np.asarray(sr_aleatorias, dtype=float)
    pool = pool[np.isfinite(pool)]
    if len(pool) < 5:
        return {"pvalor": None, "p95_mejor_azar": None, "supera": False}
    rng = np.random.default_rng(semilla)
    media, desv = float(pool.mean()), float(pool.std(ddof=1))
    n = max(int(n_pruebas), 1)
    # Máximo de n normales por inversión: F_max(x) = Φ(x)^n → x = Φ⁻¹(U^(1/n))
    u = rng.random(n_remuestreos) ** (1.0 / n)
    maximos = media + desv * np.array([ppf(v) for v in u])
    pvalor = float((maximos >= mejor_sr).mean())
    return {
        "pvalor": pvalor,
        "p95_mejor_azar": float(np.percentile(maximos, 95)),
        "mediana_mejor_azar": float(np.median(maximos)),
        "max_aleatoria_observada": float(pool.max()),
        "media_aleatorias": media,
        "desv_aleatorias": desv,
        "supera": pvalor < 0.05,
    }


def _es(x: float, dec: int = 2) -> str:
    """Número en formato español (1.234,56)."""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def explicar_dsr(dsr: float, sr_anual: float, referencia_anual: float, n_pruebas: int) -> str:
    """Explicación en lenguaje claro."""
    return (
        f"Tras probar {_es(n_pruebas, 0)} estrategias, el mejor Sharpe que cabría esperar por pura "
        f"suerte es ≈ {_es(referencia_anual)} (anualizado). Esta estrategia tiene {_es(sr_anual)} en "
        f"entrenamiento. La probabilidad de que su ventaja sea real y no suerte (DSR) es "
        f"{dsr * 100:.0f} %."
    )
