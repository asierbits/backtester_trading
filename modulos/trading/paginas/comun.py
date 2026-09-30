"""Utilidades compartidas por las páginas de Streamlit del módulo Trading."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from core import db
from core.config import cargar_config
from modulos.trading.backtest.metricas import decimal, formatear, miles  # noqa: F401 (se reexportan)
from modulos.trading.validacion.veredicto import COLORES

ICONOS_VEREDICTO = {"APROBADA": "✅", "SOSPECHOSA": "⚠️", "DESCARTADA": "⛔"}


def oscuro() -> bool:
    """¿Está la interfaz en modo oscuro? Sigue al tema de Streamlit; si no se sabe, a config.yaml."""
    try:
        tipo = st.context.theme.type
        if tipo in ("dark", "light"):
            return tipo == "dark"
    except Exception:  # noqa: BLE001
        pass
    return cargar_config().get("interfaz", {}).get("tema_graficas", "oscuro") == "oscuro"


def grafica(fig: Any, **kw: Any) -> None:
    """Muestra una figura Plotly con nuestro tema (no el de Streamlit)."""
    st.plotly_chart(fig, theme=None, width="stretch", config={"displaylogo": False, "locale": "es"}, **kw)


def etiqueta(veredicto: str | None) -> str:
    if not veredicto:
        return "—"
    return f"{ICONOS_VEREDICTO.get(veredicto, '')} {veredicto}"


def badge(veredicto: str | None) -> None:
    """Etiqueta de veredicto con color de estado + icono + texto (nunca solo color)."""
    if not veredicto:
        return
    color = COLORES.get(veredicto, "#888")
    st.markdown(
        f"<span style='background:{color}22;border:1px solid {color};border-radius:12px;"
        f"padding:3px 12px;font-weight:600'>{etiqueta(veredicto)}</span>",
        unsafe_allow_html=True,
    )


def aviso_ficticio() -> None:
    st.caption(
        "Dinero 100 % ficticio · Los resultados pasados no garantizan resultados futuros · "
        "Nada de esto es consejo financiero."
    )


def aviso_sinteticos(resumen: dict) -> None:
    if resumen.get("datos_sinteticos"):
        st.warning(
            "Este experimento usa **datos sintéticos** (generados al azar). Si alguna estrategia "
            "'gana', es por azar o por la forma del proceso aleatorio: no dice nada del mercado real.",
            icon=":material/warning:",
        )


def selector_experimento(clave: str = "experimento_id") -> dict | None:
    """Selector de experimento (recuerda la elección entre páginas)."""
    exps = db.listar_experimentos()
    exps = exps[exps["estado"] == "terminado"]
    if exps.empty:
        st.info("Todavía no hay experimentos terminados. Lanza uno en **Nuevo experimento**.")
        st.page_link(
            "modulos/trading/paginas/nuevo_experimento.py", label="Ir a Nuevo experimento", icon=":material/science:"
        )
        return None
    opciones = exps["id"].tolist()
    actual = st.session_state.get(clave)
    indice = opciones.index(actual) if actual in opciones else 0
    elegido = st.selectbox(
        "Experimento",
        opciones,
        index=indice,
        format_func=lambda i: (
            f"#{i} · {exps.set_index('id').loc[i, 'nombre']} · {exps.set_index('id').loc[i, 'creado'][:16]}"
        ),
    )
    st.session_state[clave] = elegido
    return db.obtener_experimento(int(elegido))


def pct(x: float | None) -> str:
    return formatear("rentabilidad", x)


def num(x: float | None) -> str:
    return formatear("sharpe", x)


def params_texto(p: dict | str) -> str:
    if isinstance(p, str):
        p = json.loads(p)
    return ", ".join(f"{k}={v}" for k, v in p.items())


def tabla_estado_datos() -> pd.DataFrame:
    meta = db.listar_meta_datos()
    if meta.empty:
        return meta
    calidad = meta["calidad_json"].map(json.loads)
    meta["faltantes"] = calidad.map(lambda c: c.get("velas_faltantes", 0))
    meta["% faltantes"] = calidad.map(lambda c: c.get("pct_faltantes", 0))
    meta["duplicados"] = calidad.map(lambda c: c.get("duplicados_eliminados", 0))
    meta["inválidas"] = calidad.map(lambda c: c.get("velas_invalidas_eliminadas", 0))
    meta["saltos anómalos"] = calidad.map(lambda c: c.get("saltos_anomalos", 0))
    meta["volumen cero"] = calidad.map(lambda c: c.get("volumen_cero", 0))
    return meta.drop(columns=["calidad_json", "ruta"])
