"""Inicio del módulo Trading: últimos experimentos, estado de los datos y accesos rápidos."""

import json

import streamlit as st

from core import db
from core.config import clave_anthropic
from modulos.trading.paginas import comun

st.title("Laboratorio de estrategias de trading")
st.write(
    "Aquí se prueban miles de estrategias con dinero **ficticio** para descubrir cuáles son "
    "**humo** y cuáles sobreviven a una validación seria. Un laboratorio que dice "
    "*«esto es humo»* vale más que uno que enseña gráficas bonitas."
)
comun.aviso_ficticio()

c1, c2, c3, c4 = st.columns(4)
c1.page_link("modulos/trading/paginas/datos.py", label="Datos", icon=":material/database:")
c2.page_link("modulos/trading/paginas/nuevo_experimento.py", label="Nuevo experimento", icon=":material/science:")
c3.page_link("modulos/trading/paginas/ranking.py", label="Ranking", icon=":material/leaderboard:")
c4.page_link("modulos/trading/paginas/humo.py", label="Laboratorio de humo", icon=":material/local_fire_department:")

st.subheader("Últimos experimentos")
exps = db.listar_experimentos()
if exps.empty:
    st.info("Aún no hay experimentos. Descarga datos (o genera datos de ejemplo) y lanza el primero.")
else:
    filas = []
    for e in exps.head(10).itertuples():
        r = json.loads(e.resumen_json) if e.resumen_json else {}
        conteo = r.get("conteo", {})
        filas.append(
            {
                "#": e.id,
                "Nombre": e.nombre,
                "Creado": e.creado[:16].replace("T", " "),
                "Estado": e.estado,
                "Estrategias": e.n_estrategias,
                "Aprobadas": conteo.get("APROBADA", 0),
                "Sospechosas": conteo.get("SOSPECHOSA", 0),
                "Descartadas": conteo.get("DESCARTADA", 0),
                "Datos": "sintéticos" if r.get("datos_sinteticos") else r.get("fuente", "—"),
                "Duración (s)": round(e.duracion_s or 0),
            }
        )
    st.dataframe(filas, hide_index=True, width="stretch")

st.subheader("Estado de los datos")
meta = comun.tabla_estado_datos()
if meta.empty:
    st.info("No hay datos descargados todavía. Ve a **Datos**.")
else:
    st.dataframe(
        meta[["fuente", "activo", "temporalidad", "desde", "hasta", "n_velas", "faltantes", "actualizado"]],
        hide_index=True,
        width="stretch",
    )

st.subheader("Agentes de IA")
if clave_anthropic():
    st.success(
        "Clave de Anthropic configurada: el panel de agentes críticos está disponible.", icon=":material/check_circle:"
    )
else:
    st.warning(
        "No hay clave de Anthropic en `.env`: los experimentos funcionan igual, pero sin el panel de "
        "agentes críticos. Puedes añadirla en **Ajustes**.",
        icon=":material/key:",
    )
