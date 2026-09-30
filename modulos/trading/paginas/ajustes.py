"""Ajustes: clave de la API, modelo de IA, costes por defecto, límites de gasto y tema."""

import streamlit as st

from core.config import cargar_config, clave_anthropic, guardar_config, guardar_en_env
from modulos.trading.agentes import prompts
from modulos.trading.paper import alertas

st.title("Ajustes")
cfg = cargar_config()

st.subheader("Clave de Anthropic")
if clave_anthropic():
    st.success("Hay una clave configurada en `.env` (no se muestra por seguridad).", icon=":material/key:")
else:
    st.warning("No hay clave. Sin ella, todo funciona salvo el panel de agentes.", icon=":material/key_off:")
with st.form("clave"):
    nueva = st.text_input(
        "Nueva ANTHROPIC_API_KEY", type="password", help="Se guarda solo en el archivo .env, que está en .gitignore."
    )
    if st.form_submit_button("Guardar clave") and nueva.strip():
        guardar_en_env("ANTHROPIC_API_KEY", nueva)
        st.success("Clave guardada en .env.")

with st.form("ajustes"):
    st.subheader("Agentes de IA")
    a = cfg["agentes"]
    c1, c2, c3 = st.columns(3)
    a["activado"] = c1.toggle("Panel de agentes activado por defecto", a["activado"])
    a["modelo"] = c2.text_input("Modelo", a["modelo"], help="Nombre exacto del modelo de la API de Anthropic.")
    esfuerzos = ["low", "medium", "high", "xhigh", "max"]
    a["esfuerzo"] = c3.selectbox("Esfuerzo", esfuerzos, index=esfuerzos.index(a.get("esfuerzo", "medium")))
    c1, c2, c3 = st.columns(3)
    a["concurrencia"] = int(c1.number_input("Llamadas simultáneas", 1, 10, int(a["concurrencia"])))
    a["max_llamadas_por_experimento"] = int(
        c2.number_input("Máx. llamadas por experimento", 1, 1000, int(a["max_llamadas_por_experimento"]))
    )
    a["max_tokens_por_experimento"] = int(
        c3.number_input(
            "Máx. tokens por experimento", 1000, 50_000_000, int(a["max_tokens_por_experimento"]), step=50_000
        )
    )
    c1, c2, c3 = st.columns(3)
    a["precio_entrada_millon"] = float(
        c1.number_input("$ por millón de tokens de entrada", 0.0, 100.0, float(a["precio_entrada_millon"]))
    )
    a["precio_salida_millon"] = float(
        c2.number_input("$ por millón de tokens de salida", 0.0, 500.0, float(a["precio_salida_millon"]))
    )
    a["usar_fallbacks"] = c3.toggle("Modelo de respaldo si hay rechazo", a.get("usar_fallbacks", True))
    a["panel"] = st.multiselect(
        "Agentes del panel", list(prompts.AGENTES), default=a["panel"], format_func=prompts.nombre
    )
    a["arbitro"] = st.toggle("Árbitro final", a.get("arbitro", True))
    st.caption(
        f"Versión de los prompts: {prompts.VERSION_PROMPTS} (se editan en `modulos/trading/agentes/prompts.py`)."
    )

    st.subheader("Costes y capital por defecto")
    b = cfg["backtest"]
    c1, c2, c3 = st.columns(3)
    b["capital_inicial"] = float(c1.number_input("Capital inicial (€)", 100.0, 1e7, float(b["capital_inicial"])))
    b["comision"] = c2.number_input("Comisión (%)", 0.0, 2.0, b["comision"] * 100, format="%.3f") / 100
    b["slippage"] = c3.number_input("Slippage (%)", 0.0, 2.0, b["slippage"] * 100, format="%.3f") / 100

    st.subheader("Reglas del veredicto")
    v = cfg["veredicto"]
    c1, c2, c3, c4 = st.columns(4)
    v["min_operaciones"] = int(c1.number_input("Mín. operaciones (train)", 1, 1000, int(v["min_operaciones"])))
    v["min_operaciones_test"] = int(c2.number_input("Mín. operaciones (test)", 1, 1000, int(v["min_operaciones_test"])))
    v["max_degradacion_sharpe"] = float(
        c3.slider("Sharpe de test mínimo (fracción del de train)", 0.0, 1.0, float(v["max_degradacion_sharpe"]))
    )
    v["percentil_aleatorias"] = int(
        c4.slider("Percentil de las aleatorias a superar", 50, 99, int(v["percentil_aleatorias"]))
    )

    st.subheader("Paper trading y apariencia")
    p = cfg.setdefault("paper", {})
    c1, c2, c3 = st.columns(3)
    p["alertas"] = c1.toggle(
        "Enviar alertas",
        bool(p.get("alertas")),
        help=f"Canales configurados: {', '.join(alertas.canales_configurados()) or 'ninguno'}",
    )
    p["intervalo_minutos"] = int(
        c2.number_input("Minutos entre actualizaciones (bucle)", 1, 1440, int(p.get("intervalo_minutos", 60)))
    )
    tema = cfg.setdefault("interfaz", {})
    tema["tema_graficas"] = c3.selectbox(
        "Tema de gráficas si no se detecta",
        ["oscuro", "claro"],
        index=["oscuro", "claro"].index(tema.get("tema_graficas", "oscuro")),
    )
    st.caption(
        "El tema claro/oscuro de la app se cambia en el menú ⋮ → Settings; las gráficas lo siguen automáticamente."
    )

    if st.form_submit_button("Guardar ajustes", type="primary"):
        guardar_config(cfg)
        st.success("Guardado en config.yaml.")
