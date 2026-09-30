"""Laboratorio: aplicación personal modular. Punto de entrada de Streamlit.

Arranque:  streamlit run app.py   (o los scripts arrancar.sh / arrancar.bat)
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from modulos import registro  # noqa: E402

st.set_page_config(page_title="Laboratorio", page_icon=":material/biotech:", layout="wide")


def portada() -> None:
    st.title("Laboratorio")
    st.caption("Tu aplicación personal, módulo a módulo.")
    for m in registro.descubrir():
        with st.container(border=True):
            st.subheader(m.NOMBRE)
            st.write(getattr(m, "DESCRIPCION", ""))
            primera = m.paginas()[0]
            st.page_link(primera, label=f"Abrir {m.NOMBRE}", icon=":material/arrow_forward:")
    st.info(
        "¿Quieres añadir un módulo nuevo (voz, prospección, segundo cerebro…)? Crea una carpeta en "
        "`modulos/` con un archivo `modulo.py` (NOMBRE, ORDEN y paginas()) y aparecerá en este menú.",
        icon=":material/extension:",
    )


secciones: dict[str, list] = {"": [st.Page(portada, title="Portada", icon=":material/apps:", default=True)]}
for modulo in registro.descubrir():
    secciones[modulo.NOMBRE] = modulo.paginas()

st.navigation(secciones, position="sidebar", expanded=True).run()
