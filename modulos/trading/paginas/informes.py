"""Informes: exportar un experimento completo en HTML (imprimible a PDF)."""

import streamlit as st

from core.config import ruta_informes
from modulos.trading import informes
from modulos.trading.paginas import comun

st.title("Informes")
exp = comun.selector_experimento()
if exp is None:
    st.stop()
n = st.slider("Finalistas con detalle completo", 1, 30, min(10, len(exp["resumen"].get("finalistas", [])) or 1))
if st.button("Generar informe HTML", type="primary", icon=":material/description:"):
    with st.spinner("Generando…"):
        ruta = informes.guardar(int(exp["id"]), n)
    st.success(f"Guardado en `{ruta}`")
    st.download_button("Descargar informe", ruta.read_bytes(), ruta.name, "text/html", icon=":material/download:")
st.info(
    "¿PDF? Abre el HTML en tu navegador y usa **Imprimir → Guardar como PDF**: el informe trae estilos "
    "de impresión. (Generar PDF directamente exigiría instalar un motor de renderizado pesado.)",
    icon=":material/picture_as_pdf:",
)

existentes = sorted(ruta_informes().glob("*.html"), reverse=True)
if existentes:
    st.subheader("Informes guardados")
    for r in existentes[:20]:
        c1, c2 = st.columns([4, 1])
        c1.write(r.name)
        c2.download_button("Descargar", r.read_bytes(), r.name, "text/html", key=str(r))
