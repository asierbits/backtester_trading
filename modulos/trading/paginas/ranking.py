"""Ranking: todas las estrategias con métricas train vs test y veredicto."""

import pandas as pd
import streamlit as st

from core.config import cargar_config, clave_anthropic
from modulos.trading import repositorio as repo
from modulos.trading.agentes.panel import ejecutar_panel, estimar_coste_panel
from modulos.trading.estrategias import FAMILIAS
from modulos.trading.paginas import comun

st.title("Ranking de estrategias")
exp = comun.selector_experimento()
if exp is None:
    st.stop()
comun.aviso_sinteticos(exp["resumen"])
df = repo.ranking(int(exp["id"]))
if df.empty:
    st.info("El experimento no tiene estrategias.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
veredictos = c1.multiselect(
    "Veredicto", ["APROBADA", "SOSPECHOSA", "DESCARTADA"], default=["APROBADA", "SOSPECHOSA", "DESCARTADA"]
)
familias = c2.multiselect(
    "Familia",
    sorted(df["familia"].unique()),
    format_func=lambda f: FAMILIAS[f].nombre_legible if f in FAMILIAS else f,
)
activos = c3.multiselect("Activo", sorted(df["activo"].unique()))
solo_finalistas = c4.toggle("Solo finalistas", value=False)
c1, c2 = st.columns(2)
min_ops = c1.number_input("Mínimo de operaciones en entrenamiento", 0, 10_000, 0)
orden = c2.selectbox(
    "Ordenar por",
    ["sharpe_train", "sharpe_test", "rent_test", "calmar_test", "ops_train"],
    format_func={
        "sharpe_train": "Sharpe entrenamiento",
        "sharpe_test": "Sharpe test",
        "rent_test": "Rentabilidad test",
        "calmar_test": "Calmar test",
        "ops_train": "Operaciones",
    }.get,
)

f = df[df["veredicto"].isin(veredictos)]
if familias:
    f = f[f["familia"].isin(familias)]
if activos:
    f = f[f["activo"].isin(activos)]
if solo_finalistas:
    f = f[f["finalista"] == 1]
f = f[f["ops_train"] >= min_ops].sort_values(orden, ascending=False)

st.caption(
    f"{comun.miles(len(f))} de {comun.miles(len(df))} estrategias (se muestran hasta 5.000). "
    "Pulsa una fila para ver su detalle."
)
vista = pd.DataFrame(
    {
        "id": f["id"],
        "Veredicto": f["veredicto"].map(comun.etiqueta),
        "Familia": f["familia"],
        "Parámetros": f["parametros"].map(comun.params_texto),
        "Activo": f["activo"],
        "Finalista": f["rango_train"].map(lambda r: f"#{int(r)}" if pd.notna(r) else ""),
        "Sharpe train": f["sharpe_train"],
        "Sharpe test": f["sharpe_test"],
        "Rent. train %": f["rent_train"] * 100,
        "Rent. test %": f["rent_test"] * 100,
        "Caída test %": f["dd_test"] * 100,
        "Calmar test": f["calmar_test"],
        "Ops train": f["ops_train"],
        "Ops test": f["ops_test"],
        "% gan. test": f["ganadoras_test"] * 100,
        "Motivo": f["motivos"].map(lambda m: m[0] if m else ""),
    }
).head(5000)
num = st.column_config.NumberColumn
sel = st.dataframe(
    vista,
    hide_index=True,
    width="stretch",
    height=520,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "id": num("id", format="%d"),
        "Sharpe train": num(format="%.2f"),
        "Sharpe test": num(format="%.2f"),
        "Rent. train %": num(format="%.1f"),
        "Rent. test %": num(format="%.1f"),
        "Caída test %": num(format="%.1f"),
        "Calmar test": num(format="%.2f"),
        "Ops train": num(format="%d"),
        "Ops test": num(format="%d"),
        "% gan. test": num(format="%.0f"),
        "Motivo": st.column_config.TextColumn(width="large"),
    },
)
filas = sel.selection.rows if sel and sel.selection else []
if filas:
    st.session_state["estrategia_id"] = int(vista.iloc[filas[0]]["id"])
    st.switch_page("modulos/trading/paginas/detalle.py")

st.download_button(
    "Descargar CSV",
    vista.to_csv(index=False).encode("utf-8"),
    f"ranking_{exp['id']}.csv",
    "text/csv",
    icon=":material/download:",
)

st.divider()
st.subheader("Panel de agentes críticos")
ids = repo.finalistas(int(exp["id"]))
criticadas = sum(1 for i in ids if repo.criticas(i))
st.write(f"{len(ids)} finalistas · {criticadas} ya tienen críticas.")
if not clave_anthropic():
    st.info("Añade tu ANTHROPIC_API_KEY en **Ajustes** para usar el panel de agentes.")
elif ids:
    cfg = exp["config"]
    cfg["agentes"] = cargar_config()["agentes"]  # modelo y límites actuales
    est = estimar_coste_panel(int(exp["id"]), cfg, ids)
    st.write(
        f"Estimación: {est['llamadas']} llamadas, ~{comun.miles(est['tokens_entrada'])} tokens de entrada y "
        f"~{comun.miles(est['tokens_salida'])} de salida → **~{comun.decimal(est['coste_usd'], 2)} $** con "
        f"{cfg['agentes']['modelo']}. Las respuestas ya calculadas salen de la caché sin coste."
    )
    if not est["dentro_de_limites"]:
        st.warning("La estimación supera los límites de gasto configurados: se evaluarán las que quepan.")
    if st.button("Pasar finalistas por el panel", icon=":material/smart_toy:"):
        barra = st.progress(0.0)
        with st.spinner("Los agentes están deliberando…"):
            info = ejecutar_panel(int(exp["id"]), cfg=cfg, ids=ids, progreso=lambda x, t: barra.progress(x, t))
        st.success(
            f"Hecho: {info['evaluadas']} evaluadas, {info['endurecidas']} veredictos endurecidos, "
            f"coste real ~{comun.decimal(info['coste_usd'], 2)} $."
        )
        if info.get("aviso"):
            st.warning(info["aviso"])
