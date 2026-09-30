"""Nuevo experimento: elegir activos, periodo, familias, rangos, costes… y lanzar."""

import os
import time

import streamlit as st
import yaml

from core import db
from core.config import cargar_config, clave_anthropic
from modulos.trading.estrategias import FAMILIAS
from modulos.trading.estrategias.generador import generar
from modulos.trading.experimento import ejecutar_experimento
from modulos.trading.paginas import comun

st.title("Nuevo experimento")
comun.aviso_ficticio()
cfg = cargar_config()
e, bt, pa = cfg["experimento"], cfg["backtest"], cfg["parametros"]

meta = db.listar_meta_datos()

with st.form("experimento"):
    st.subheader("Datos")
    c1, c2, c3 = st.columns(3)
    fuentes = ["sintetico", "binance", "yfinance", "csv"]
    fuente = c1.selectbox(
        "Fuente de datos",
        fuentes,
        index=fuentes.index(e.get("fuente", "sintetico")),
        help="'sintetico' genera datos de ejemplo al vuelo si no existen.",
    )
    temporalidad = c2.selectbox("Temporalidad", ["1h", "4h", "1d"], index=["1h", "4h", "1d"].index(e["temporalidad"]))
    disponibles = sorted(set(meta[meta["fuente"] == fuente]["activo"])) if not meta.empty else []
    if fuente in ("binance", "sintetico"):
        disponibles = sorted(set(disponibles) | set(cfg["datos"]["activos"]))
    elif fuente == "yfinance":
        disponibles = sorted(set(disponibles) | set(cfg["datos"].get("acciones", [])))
    activos = c3.multiselect(
        "Activos", disponibles, default=[a for a in e["activos"] if a in disponibles] or disponibles[:1]
    )
    c1, c2, c3 = st.columns(3)
    desde = c1.text_input("Desde (vacío = todo)", e.get("desde") or "")
    hasta = c2.text_input("Hasta (vacío = hasta el final)", e.get("hasta") or "")
    proporcion = c3.slider(
        "Proporción de entrenamiento",
        0.3,
        0.8,
        float(e["proporcion_train"]),
        0.05,
        help="El resto (lo más reciente) es el test, que nunca se usa para elegir.",
    )

    st.subheader("Estrategias")
    c1, c2, c3 = st.columns(3)
    familias = c1.multiselect(
        "Familias", list(FAMILIAS), default=e["familias"], format_func=lambda f: FAMILIAS[f].nombre_legible
    )
    max_est = c2.number_input(
        "Máximo de combinaciones de parámetros",
        10,
        100_000,
        int(e["max_estrategias"]),
        step=500,
        help="Cada combinación se prueba en cada activo.",
    )
    top = c3.number_input("Finalistas (validación completa + agentes)", 1, 50, int(e["top_finalistas"]))
    textos_param: dict[str, str] = {}
    with st.expander("Rangos de parámetros ([mín, máx, paso] o lista de opciones)"):
        for f in FAMILIAS:
            if f in pa:
                textos_param[f] = st.text_area(
                    FAMILIAS[f].nombre_legible,
                    yaml.safe_dump(pa[f], allow_unicode=True, sort_keys=False, default_flow_style=None),
                    key=f"par_{f}",
                    height=110,
                )

    st.subheader("Costes y capital")
    c1, c2, c3, c4 = st.columns(4)
    capital = c1.number_input("Capital inicial (€ ficticios)", 100.0, 1e7, float(bt["capital_inicial"]), step=100.0)
    comision = c2.number_input("Comisión por operación (%)", 0.0, 2.0, bt["comision"] * 100, step=0.01, format="%.3f")
    slippage = c3.number_input("Slippage (%)", 0.0, 2.0, bt["slippage"] * 100, step=0.01, format="%.3f")
    modos = ["fijo", "volatilidad", "volumen"]
    modo_slip = c4.selectbox("Slippage", modos, index=modos.index(bt.get("modo_slippage", "fijo")))
    c1, c2, c3, c4 = st.columns(4)
    ejecucion = c1.selectbox(
        "Ejecución de la señal",
        ["apertura", "cierre"],
        index=["apertura", "cierre"].index(bt.get("precio_ejecucion", "apertura")),
        help="La señal del cierre de t se ejecuta en la apertura (o el cierre) de t+1.",
    )
    cortos = c2.toggle("Permitir cortos (-1)", bool(bt.get("permitir_cortos")))
    sizing = c3.selectbox(
        "Tamaño de posición",
        ["completo", "volatilidad"],
        index=["completo", "volatilidad"].index(bt.get("sizing", "completo")),
        format_func={"completo": "100 % del capital", "volatilidad": "Por volatilidad"}.get,
    )
    vol_obj = c4.number_input(
        "Volatilidad objetivo anual", 0.05, 2.0, float(bt.get("volatilidad_objetivo", 0.5)), step=0.05
    )

    st.subheader("Control y reproducibilidad")
    c1, c2, c3 = st.columns(3)
    semilla = c1.number_input("Semilla aleatoria", 0, 2**31 - 1, int(cfg["general"]["semilla"]))
    n_al = c2.number_input("Estrategias aleatorias de control por activo", 20, 5000, int(e["n_aleatorias"]))
    agentes = c3.toggle(
        "Pasar finalistas por el panel de agentes",
        bool(cfg["agentes"]["activado"]) and bool(clave_anthropic()),
        disabled=not clave_anthropic(),
        help="Necesita ANTHROPIC_API_KEY en .env",
    )
    nombre = st.text_input("Nombre del experimento (opcional)")
    st.form_submit_button("Calcular estimación", icon=":material/calculate:")

errores = []
parametros = dict(pa)
for f, texto in textos_param.items():
    try:
        parametros[f] = yaml.safe_load(texto)
    except yaml.YAMLError as err:
        errores.append(f"Rangos de {f}: {err}")
if not activos:
    errores.append("Elige al menos un activo.")
if not familias:
    errores.append("Elige al menos una familia de estrategias.")
for err in errores:
    st.error(err)
if errores:
    st.stop()

cambios = {
    "general": {"semilla": int(semilla)},
    "experimento": {
        "fuente": fuente,
        "temporalidad": temporalidad,
        "activos": activos,
        "desde": desde or None,
        "hasta": hasta or None,
        "max_estrategias": int(max_est),
        "familias": familias,
        "proporcion_train": float(proporcion),
        "n_aleatorias": int(n_al),
        "top_finalistas": int(top),
    },
    "backtest": {
        "capital_inicial": float(capital),
        "comision": comision / 100,
        "slippage": slippage / 100,
        "modo_slippage": modo_slip,
        "precio_ejecucion": ejecucion,
        "permitir_cortos": bool(cortos),
        "sizing": sizing,
        "volatilidad_objetivo": float(vol_obj),
    },
    "parametros": parametros,
}

try:
    gen = generar(familias, parametros, int(max_est), int(semilla))
except Exception as err:  # noqa: BLE001
    st.error(f"Los rangos de parámetros no son válidos: {err}")
    st.stop()

velas_ano = {"1h": 8766, "4h": 2191, "1d": 365}[temporalidad]
anos = 6.5
if not meta.empty:
    sub = meta[(meta["fuente"] == fuente) & (meta["temporalidad"] == temporalidad) & (meta["activo"].isin(activos))]
    if len(sub):
        anos = sub["n_velas"].mean() / velas_ano
total = gen.n * len(activos)
nucleos = os.cpu_count() or 1
segundos = 8 + total * anos * velas_ano * 6e-8 / nucleos + int(top) * 3
c1, c2, c3 = st.columns(3)
c1.metric(
    "Estrategias a probar",
    comun.miles(total),
    f"{comun.miles(gen.n)} combinaciones × {len(activos)} activos",
    delta_color="off",
    delta_arrow="off",
)
c2.metric("Espacio total de parámetros", comun.miles(gen.espacio_total))
c3.metric(
    "Tiempo estimado",
    f"~{comun.decimal(segundos / 60)} min" if segundos > 90 else f"~{segundos:.0f} s",
    f"{nucleos} núcleos",
    delta_color="off",
    delta_arrow="off",
)
st.caption(
    ", ".join(
        f"{FAMILIAS[f].nombre_legible}: {comun.miles(v['probadas'])}/{comun.miles(v['espacio'])}"
        for f, v in gen.por_familia.items()
    )
)
if agentes:
    ca = cfg["agentes"]
    llamadas = int(top) * (len(ca["panel"]) + int(bool(ca.get("arbitro", True))))
    coste = (
        llamadas
        * (9000 * ca["precio_entrada_millon"] + ca["tokens_salida_estimados"] * ca["precio_salida_millon"])
        / 1e6
    )
    st.info(
        f"Panel de agentes: ~{llamadas} llamadas a {ca['modelo']}, coste aproximado "
        f"**~{comun.decimal(coste, 2)} $** (límite: {ca['max_llamadas_por_experimento']} llamadas y "
        f"{comun.miles(ca['max_tokens_por_experimento'])} tokens por experimento).",
        icon=":material/smart_toy:",
    )

if st.button("Lanzar experimento", type="primary", icon=":material/rocket_launch:"):
    barra = st.progress(0.0, "Preparando…")
    inicio = time.time()

    def progreso(fraccion: float, texto: str) -> None:
        barra.progress(fraccion, f"{texto} · {time.time() - inicio:.0f} s")

    try:
        ident = ejecutar_experimento(cambios, nombre or None, progreso, lanzar_agentes=bool(agentes))
    except Exception as err:  # noqa: BLE001
        st.error(f"El experimento falló: {err}")
        if fuente != "sintetico":
            st.info("¿Faltan datos? Descárgalos en **Datos** o usa la fuente 'sintetico' para probar.")
    else:
        st.session_state["experimento_id"] = ident
        exp = db.obtener_experimento(ident)
        conteo = exp["resumen"].get("conteo", {})
        st.success(
            f"Experimento #{ident} terminado en {exp['duracion_s']:.0f} s: "
            f"{conteo.get('APROBADA', 0)} aprobadas, {conteo.get('SOSPECHOSA', 0)} sospechosas y "
            f"{comun.miles(conteo.get('DESCARTADA', 0))} descartadas."
        )
        comun.aviso_sinteticos(exp["resumen"])
        if "agentes" in exp["resumen"]:
            st.json(exp["resumen"]["agentes"], expanded=False)
        c1, c2 = st.columns(2)
        c1.page_link("modulos/trading/paginas/ranking.py", label="Ver ranking", icon=":material/leaderboard:")
        c2.page_link(
            "modulos/trading/paginas/humo.py", label="Ver laboratorio de humo", icon=":material/local_fire_department:"
        )
