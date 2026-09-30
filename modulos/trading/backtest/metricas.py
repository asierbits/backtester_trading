"""Métricas de rendimiento y riesgo.

Se calculan en numba para poder evaluar miles de estrategias en segundos. Convenciones:
- Rentabilidades por vela a partir de la curva de capital (mark-to-market).
- Sharpe y Sortino anualizados con tipo libre de riesgo 0 y √(velas por año).
- Caída máxima (max drawdown) en fracción positiva (0,25 = −25 %).
- Las operaciones incluyen la abierta al final (valorada a mercado).
"""

from __future__ import annotations

import numpy as np
from numba import njit

CLAVES_METRICAS = [
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
]

NOMBRES_METRICAS = {
    "rentabilidad": "Rentabilidad total",
    "cagr": "Rentabilidad anualizada (CAGR)",
    "volatilidad": "Volatilidad anualizada",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_dd": "Caída máxima",
    "dur_max_dd_dias": "Duración peor caída (días)",
    "calmar": "Calmar",
    "n_ops": "Nº operaciones",
    "pct_ganadoras": "% ganadoras",
    "ratio_ganancia_perdida": "Ganancia media / pérdida media",
    "profit_factor": "Profit factor",
    "expectativa": "Expectativa por operación",
    "exposicion": "Exposición al mercado",
    "racha_perdidas": "Mayor racha de pérdidas",
}

# Métricas que se muestran en porcentaje
PORCENTUALES = {"rentabilidad", "cagr", "volatilidad", "max_dd", "pct_ganadoras", "expectativa", "exposicion"}

TOPE = 100.0  # Para profit factor / Calmar sin pérdidas (evita infinitos)


@njit(cache=True)
def resumen_numba(
    equity: np.ndarray,
    capital: float,
    pct_ops: np.ndarray,
    n_ops: int,
    expuestas: int,
    velas_ano: float,
) -> np.ndarray:
    """Calcula todas las métricas (orden de CLAVES_METRICAS)."""
    salida = np.zeros(15)
    n = equity.shape[0]
    if n == 0:
        return salida
    # Rentabilidades por vela
    suma = 0.0
    suma2 = 0.0
    suma_neg2 = 0.0
    previo = capital
    pico = capital
    max_dd = 0.0
    dur = 0
    max_dur = 0
    for i in range(n):
        r = equity[i] / previo - 1.0 if previo > 0 else 0.0
        suma += r
        suma2 += r * r
        if r < 0:
            suma_neg2 += r * r
        previo = equity[i]
        if equity[i] >= pico:
            pico = equity[i]
            dur = 0
        else:
            dur += 1
            dd = 1.0 - equity[i] / pico
            if dd > max_dd:
                max_dd = dd
        if dur > max_dur:
            max_dur = dur
    media = suma / n
    var = suma2 / n - media * media
    desv = np.sqrt(var) if var > 0 else 0.0
    desv_neg = np.sqrt(suma_neg2 / n)
    final = equity[n - 1]
    rent = final / capital - 1.0
    anos = n / velas_ano
    if final <= 0:
        cagr = -1.0
    elif anos > 0:
        cagr = (final / capital) ** (1.0 / anos) - 1.0
    else:
        cagr = 0.0
    raiz = np.sqrt(velas_ano)
    salida[0] = rent
    salida[1] = cagr
    salida[2] = desv * raiz
    salida[3] = media / desv * raiz if desv > 1e-12 else 0.0
    if desv_neg > 1e-12:
        salida[4] = media / desv_neg * raiz
    else:
        salida[4] = TOPE if media > 0 else 0.0
    salida[5] = max_dd
    salida[6] = max_dur * 365.0 / velas_ano
    if max_dd > 1e-9:
        salida[7] = min(cagr / max_dd, TOPE)
    else:
        salida[7] = TOPE if cagr > 0 else 0.0

    # Operaciones
    ganadoras = 0
    suma_g = 0.0
    suma_p = 0.0
    racha = 0
    peor_racha = 0
    suma_pct = 0.0
    for j in range(n_ops):
        x = pct_ops[j]
        suma_pct += x
        if x > 0:
            ganadoras += 1
            suma_g += x
            racha = 0
        else:
            suma_p += -x
            racha += 1
            if racha > peor_racha:
                peor_racha = racha
    perdedoras = n_ops - ganadoras
    salida[8] = n_ops
    salida[9] = ganadoras / n_ops if n_ops > 0 else 0.0
    if ganadoras > 0 and perdedoras > 0 and suma_p > 0:
        salida[10] = (suma_g / ganadoras) / (suma_p / perdedoras)
    elif ganadoras > 0:
        salida[10] = TOPE
    salida[11] = min(suma_g / suma_p, TOPE) if suma_p > 1e-12 else (TOPE if suma_g > 0 else 0.0)
    salida[12] = suma_pct / n_ops if n_ops > 0 else 0.0
    salida[13] = expuestas / n
    salida[14] = peor_racha
    return salida


def rentabilidades_por_vela(equity: np.ndarray, capital: float) -> np.ndarray:
    """Rentabilidad de cada vela a partir de la curva de capital."""
    previo = np.concatenate([[capital], equity[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(previo > 0, equity / previo - 1.0, 0.0)
    return r


def formatear(clave: str, valor: float | None) -> str:
    """Texto legible en español (coma decimal)."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "—"
    if clave in PORCENTUALES:
        return f"{valor * 100:,.1f} %".replace(",", "X").replace(".", ",").replace("X", ".")
    if clave in {"n_ops", "racha_perdidas", "dur_max_dd_dias"}:
        return f"{int(round(valor))}"
    return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def miles(n: float | int | None) -> str:
    """Entero con punto de miles: 25000 → '25.000'."""
    if n is None:
        return "—"
    return f"{int(round(n)):,}".replace(",", ".")


def decimal(x: float | None, dec: int = 1) -> str:
    """Número con coma decimal y punto de miles: 1234.5 → '1.234,5'."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
