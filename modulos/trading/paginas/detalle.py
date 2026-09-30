"""Detalle de una estrategia: curvas, operaciones, validaciones y críticas de los agentes."""

import pandas as pd
import streamlit as st

from core.config import cargar_config, clave_anthropic
from modulos.trading import analisis, graficas
from modulos.trading import repositorio as repo
from modulos.trading.backtest.metricas import NOMBRES_METRICAS, formatear
from modulos.trading.paginas import comun

st.title("Detalle de estrategia")
exp = comun.selector_experimento()
if exp is None:
    st.stop()
rk = repo.ranking(int(exp["id"]))
if rk.empty:
    st.stop()

ids = rk["id"].tolist()
actual = st.session_state.get("estrategia_id")
if actual not in ids:
    actual = (
        int(rk[rk["finalista"] == 1].sort_values("rango_train")["id"].iloc[0])
        if (rk["finalista"] == 1).any()
        else ids[0]
    )
etiquetas = rk.set_index("id")
elegido = st.selectbox(
    "Estrategia",
    ids,
    index=ids.index(actual),
    format_func=lambda i: (
        f"#{i} · {etiquetas.loc[i, 'familia']} · {comun.params_texto(etiquetas.loc[i, 'parametros'])} · "
        f"{etiquetas.loc[i, 'activo']} · {etiquetas.loc[i, 'veredicto']}"
    ),
)
st.session_state["estrategia_id"] = int(elegido)
oscuro = comun.oscuro()

with st.spinner("Recalculando el backtest…"):
    d = analisis.detalle(int(elegido))
info = d.info
comun.aviso_sinteticos(exp["resumen"])

st.subheader(f"{d.nombre} · {info['activo']} {info['temporalidad']}")
comun.badge(info.get("veredicto"))
if info.get("veredicto_automatico") and info.get("veredicto") != info.get("veredicto_automatico"):
    st.caption(f"Veredicto automático: {info['veredicto_automatico']} · endurecido por el panel de agentes.")
for m in info.get("motivos", []):
    st.markdown(f"- {m}")

# ------------------------------------------------------------------ métricas
res = info.get("resultados", {})
bh = exp["resumen"]["activos"][info["activo"]]
claves = [
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
tabla = pd.DataFrame(
    {
        "Métrica": [NOMBRES_METRICAS[k] for k in claves] + ["Rentabilidad bruta (sin costes)"],
        "Entrenamiento": [formatear(k, res.get("train", {}).get(k)) for k in claves]
        + [formatear("rentabilidad", res.get("train", {}).get("rentabilidad_bruta"))],
        "Test": [formatear(k, res.get("test", {}).get(k)) for k in claves]
        + [formatear("rentabilidad", res.get("test", {}).get("rentabilidad_bruta"))],
        "Test costes x2": [formatear(k, res.get("test_x2", {}).get(k)) for k in claves] + ["—"],
        "Comprar y mantener (test)": [formatear(k, bh["bh_test"].get(k)) for k in claves] + ["—"],
    }
)
st.dataframe(tabla, hide_index=True, width="stretch", height=38 * 17)
st.caption(
    f"Entrenamiento: {bh['desde'][:10]} → {bh['inicio_test'][:10]} · Test: {bh['inicio_test'][:10]} → {bh['hasta'][:10]}. "
    "La estrategia se eligió solo con el entrenamiento."
)

# ------------------------------------------------------------------ gráficas
comun.grafica(graficas.curva_capital(d.resultado.equity, d.buy_hold.equity, oscuro, d.fecha_corte))
comun.grafica(graficas.caida(d.resultado.equity, oscuro, d.fecha_corte))
ops = d.resultado.operaciones
comun.grafica(graficas.precio_con_operaciones(d.ctx.mercado.df, ops, oscuro))
if len(ops):
    comun.grafica(graficas.distribucion_operaciones(ops, oscuro))
    with st.expander(f"Operaciones ({len(ops)})"):
        vista = ops.copy()
        vista["resultado_pct"] = vista["resultado_pct"] * 100
        vista["tramo"] = (vista["entrada"] >= d.fecha_corte).map({True: "test", False: "train"})
        st.dataframe(vista, hide_index=True, width="stretch")

val = info.get("validaciones", {})
pestanas = st.tabs(
    ["Mapa de calor", "Walk-forward", "Monte Carlo", "Robustez", "Múltiples pruebas", "Regímenes", "Agentes"]
)

with pestanas[0]:
    numericos = analisis.parametros_numericos(d)
    if len(numericos) < 2:
        st.info("Esta estrategia tiene menos de dos parámetros numéricos.")
    else:
        c1, c2, c3 = st.columns(3)
        px = c1.selectbox("Eje X", numericos, index=0)
        py = c2.selectbox("Eje Y", [p for p in numericos if p != px], index=0)
        tam = c3.slider("Valores por eje", 5, 25, 12)
        with st.spinner("Calculando el mapa de calor…"):
            mapa = analisis.mapa_calor(d, px, py, tam)
        comun.grafica(graficas.mapa_calor(mapa, oscuro, (info["parametros"].get(px), info["parametros"].get(py))))
        st.caption(
            "Zonas amplias con Sharpe parecido (mesetas) = ventaja más creíble. Un punto brillante aislado "
            "rodeado de malos resultados = sobreajuste. El recuadro marca los parámetros de la estrategia "
            "(si no aparece, su valor cae entre dos celdas del mapa)."
        )

if not val:
    for p in pestanas[1:6]:
        with p:
            st.info("Esta estrategia no fue finalista, así que no pasó la validación completa.")
    with pestanas[1]:
        if st.button("Validar a fondo esta estrategia", icon=":material/fact_check:"):
            from modulos.trading.experimento import validar_a_fondo

            with st.spinner("Walk-forward, Monte Carlo, robustez, múltiples pruebas…"):
                v, _ = validar_a_fondo(int(elegido))
            st.success(f"Veredicto: {v}")
            st.rerun()
else:
    with pestanas[1]:
        wf = val.get("walk_forward")
        if wf:
            c1, c2, c3 = st.columns(3)
            c1.metric("Ventanas rentables (re-optimizando)", f"{wf['pct_ventanas_rentables'] * 100:.0f} %")
            c2.metric("Ventanas rentables (parámetros fijos)", f"{wf['pct_ventanas_rentables_fijos'] * 100:.0f} %")
            c3.metric("Rentabilidad encadenada fuera de muestra", comun.pct(wf["rentabilidad_oos_total"]))
            comun.grafica(graficas.walk_forward(wf, oscuro))
            st.dataframe(
                pd.DataFrame(wf["ventanas"]).assign(
                    parametros_elegidos=lambda x: x["parametros_elegidos"].map(comun.params_texto)
                ),
                hide_index=True,
                width="stretch",
            )
    with pestanas[2]:
        mc = val.get("monte_carlo", {})
        if mc.get("valido"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rentabilidad p5", comun.pct(mc["rentabilidad_p5"]))
            c2.metric("Rentabilidad mediana", comun.pct(mc["rentabilidad_p50"]))
            c3.metric("Caída máxima p95", comun.pct(mc["max_dd_p95"]))
            c4.metric("Probabilidad de perder", f"{mc['prob_perdida'] * 100:.0f} %")
            comun.grafica(graficas.abanico_monte_carlo(mc, oscuro))
        else:
            st.info(mc.get("motivo", "Sin datos de Monte Carlo."))
    with pestanas[3]:
        rb = val.get("robustez", {})
        if rb.get("valido"):
            c1, c2 = st.columns(2)
            c1.metric("Vecinos con Sharpe > 0", f"{rb['frac_positivas'] * 100:.0f} %")
            c2.metric("Sharpe mediano de los vecinos / original", f"{rb['ratio_mediana'] * 100:.0f} %")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Parámetros": comun.params_texto(v["parametros"]), "Sharpe train": v["sharpe"]}
                        for v in rb["vecinos"]
                    ]
                ).sort_values("Sharpe train"),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info(rb.get("motivo", "Sin prueba de robustez."))
    with pestanas[4]:
        mp = val.get("multiples_pruebas", {})
        if mp:
            st.write(mp["explicacion"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Deflated Sharpe Ratio", f"{mp['dsr'] * 100:.0f} %")
            c2.metric("Sharpe máx. esperado por suerte", comun.num(mp["sharpe_referencia_anual"]))
            c3.metric("Benjamini-Hochberg", "significativa" if mp["benjamini_hochberg"] else "no significativa")
            rc = mp.get("reality_check", {})
            if rc.get("pvalor") is not None:
                st.caption(
                    f"Reality check del experimento: el mejor Sharpe real ({comun.num(rc.get('mejor_sharpe_real'))}) frente "
                    f"al percentil 95 del mejor de {comun.miles(mp['n_pruebas'])} estrategias aleatorias "
                    f"({comun.num(rc['p95_mejor_azar'])}); p-valor {comun.decimal(rc['pvalor'], 3)}."
                )
    with pestanas[5]:
        rg = val.get("regimenes", {})
        if rg:
            filas = []
            for nombre, datos in rg["regimenes"].items():
                if datos.get("velas"):
                    filas.append(
                        {
                            "Régimen": nombre,
                            "% del tiempo": datos["pct_tiempo"] * 100,
                            "Estrategia (anual.) %": datos["rent_anual_estrategia"] * 100,
                            "Comprar y mantener (anual.) %": datos["rent_anual_bh"] * 100,
                        }
                    )
            st.dataframe(pd.DataFrame(filas).round(1), hide_index=True, width="stretch")
            st.caption("Régimen según la rentabilidad del activo en los 90 días anteriores (±10 %).")

with pestanas[6]:
    criticas = info.get("criticas", [])
    if not criticas:
        st.info("Esta estrategia aún no tiene críticas de los agentes.")
    for c in criticas:
        with st.container(border=True):
            st.markdown(
                f"**{c.get('agente')}** · puntuación **{c.get('puntuacion')}/10** · recomienda **{c.get('recomendacion')}**"
            )
            st.write(c.get("explicacion"))
            col1, col2 = st.columns(2)
            col1.markdown("**Preocupaciones**\n" + "\n".join(f"- {x}" for x in c.get("preocupaciones", [])))
            col2.markdown("**Puntos fuertes**\n" + "\n".join(f"- {x}" for x in c.get("puntos_fuertes", [])))
            if c.get("prueba_adicional"):
                st.markdown(f"*Prueba adicional sugerida:* {c['prueba_adicional']}")
    if clave_anthropic() and val and st.button("Pasar esta estrategia por el panel", icon=":material/smart_toy:"):
        from modulos.trading.agentes.panel import ejecutar_panel

        cfg = exp["config"]
        cfg["agentes"] = cargar_config()["agentes"]
        with st.spinner("Los agentes están deliberando…"):
            out = ejecutar_panel(int(exp["id"]), cfg=cfg, ids=[int(elegido)])
        st.success(f"Hecho. Coste ~{comun.decimal(out['coste_usd'], 3)} $.")
        st.rerun()

st.divider()
st.subheader("Paper trading")
if info.get("veredicto") != "APROBADA":
    st.caption("Recomendación del laboratorio: seguir en paper trading solo las estrategias APROBADAS.")
c1, c2 = st.columns(2)
fuente_paper = c1.selectbox(
    "Fuente de precios",
    ["binance", exp["config"]["experimento"]["fuente"]],
    key="fuente_paper",
    help="Binance = precios reales actuales. La fuente del experimento sirve para una demostración sin conexión.",
)
if c2.button("Seguir esta estrategia en paper trading", icon=":material/monitoring:"):
    from modulos.trading.paper import paper_trading

    pid = paper_trading.registrar(int(elegido), fuente=fuente_paper)
    st.success(f"Alta en paper trading (seguimiento #{pid}). Actualízala en la página Paper trading.")
comun.aviso_ficticio()
