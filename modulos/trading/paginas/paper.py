"""Paper trading (Fase 3): seguir estrategias sobre precios actuales con dinero ficticio."""

import pandas as pd
import streamlit as st

from core.config import cargar_config
from modulos.trading import graficas
from modulos.trading import repositorio as repo
from modulos.trading.paginas import comun
from modulos.trading.paper import alertas, cartera, paper_trading

st.title("Paper trading")
st.write(
    "Sigue estrategias con **dinero ficticio** sobre precios actuales, con el mismo motor que el "
    "backtest, y compara lo que pasa con lo que predijo. Nunca se envían órdenes a ningún broker."
)
comun.aviso_ficticio()
oscuro = comun.oscuro()
cfg = cargar_config()

canales = alertas.canales_configurados()
st.caption(
    "Alertas: "
    + (", ".join(canales) if canales else "ninguna configurada (Telegram o email en .env)")
    + (" · activadas" if cfg.get("paper", {}).get("alertas") else " · desactivadas en Ajustes")
)

tabla = paper_trading.listar()
if tabla.empty:
    st.info(
        "No hay estrategias en seguimiento. Da de alta una desde **Detalle de estrategia** (idealmente una APROBADA)."
    )
else:
    if st.button("Actualizar todas ahora", type="primary", icon=":material/refresh:"):
        with st.spinner("Descargando precios y recalculando señales…"):
            resultados = paper_trading.actualizar_todas()
        for pid, est in resultados.items():
            if "error" in est:
                st.error(f"#{pid}: {est['error']}")
            for a in est.get("alertas", []):
                st.toast(a)
        tabla = paper_trading.listar()
    st.caption("Para que se actualice solo: `python -m modulos.trading.paper.paper_trading --bucle` en una terminal.")

    for fila in tabla.itertuples():
        estado = fila.estado or {}
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([2.4, 1.3, 1, 1])
            c1.markdown(f"**#{fila.id} · {fila.nombre}** ({fila.temporalidad}, precios: {fila.fuente})")
            c1.caption(f"Desde {fila.inicio[:16]} · última actualización {fila.ultima_actualizacion or 'nunca'}")
            c2.metric(
                "Capital",
                f"{comun.decimal(estado.get('equity', fila.capital), 0)} €",
            )
            c3.metric("Rentabilidad", comun.pct(estado.get("rentabilidad")))
            posicion = {1: "Largo", -1: "Corto", 0: "Fuera"}.get(estado.get("posicion", 0), "—")
            c4.metric(
                "Posición",
                posicion,
                f"próxima vela: {({1: 'largo', -1: 'corto', 0: 'fuera'}).get(estado.get('senal_siguiente_vela'), '—')}",
                delta_color="off",
                delta_arrow="off",
            )
            if estado.get("mensaje"):
                st.info(estado["mensaje"])
            with st.expander("Detalle"):
                curva = estado.get("curva")
                if curva and curva.get("fechas"):
                    serie = pd.Series(curva["equity"], index=pd.to_datetime(curva["fechas"]))
                    comun.grafica(graficas.curva_capital(serie, None, oscuro, None, "Capital en paper trading"))
                st.markdown("**Backtest (test) vs paper en vivo**")
                comp = paper_trading.comparar_con_backtest(int(fila.id))
                st.dataframe(comp.round(3), width="stretch")
                ops = paper_trading.operaciones(int(fila.id))
                if len(ops):
                    st.dataframe(ops.drop(columns=["paper_id"]), hide_index=True, width="stretch")
                b1, b2, b3 = st.columns(3)
                if b1.button("Actualizar", key=f"act_{fila.id}"):
                    try:
                        paper_trading.actualizar(int(fila.id))
                    except Exception as e:  # noqa: BLE001
                        st.error(str(e))
                    st.rerun()
                if b2.button("Pausar" if fila.activa else "Reanudar", key=f"pau_{fila.id}"):
                    paper_trading.pausar(int(fila.id), not bool(fila.activa))
                    st.rerun()
                if b3.button("Eliminar", key=f"del_{fila.id}"):
                    paper_trading.borrar(int(fila.id))
                    st.rerun()

st.divider()
st.subheader("Cartera entre estrategias")
st.write(
    "Combina varias estrategias (normalmente las APROBADAS). Los pesos se calculan con el tramo de "
    "entrenamiento y la cartera se evalúa en el de test."
)
exp = comun.selector_experimento("experimento_cartera")
if exp is not None:
    rk = repo.ranking(int(exp["id"]))
    candidatas = rk[rk["veredicto"].isin(["APROBADA", "SOSPECHOSA"]) & (rk["finalista"] == 1)]
    if candidatas.empty:
        candidatas = rk[rk["finalista"] == 1]
    opciones = candidatas["id"].tolist()
    etiquetas = candidatas.set_index("id")
    elegidas = st.multiselect(
        "Estrategias",
        opciones,
        default=candidatas[candidatas["veredicto"] == "APROBADA"]["id"].tolist()[:5],
        format_func=lambda i: (
            f"#{i} {etiquetas.loc[i, 'familia']} {etiquetas.loc[i, 'activo']} · {etiquetas.loc[i, 'veredicto']}"
        ),
    )
    metodo = st.radio("Reparto", list(cartera.METODOS), format_func=cartera.METODOS.get, horizontal=True)
    if len(elegidas) >= 2 and st.button("Construir cartera", icon=":material/pie_chart:"):
        try:
            with st.spinner("Calculando…"):
                c = cartera.construir_cartera(elegidas, metodo)
        except ValueError as e:
            st.error(str(e))
        else:
            m = c["metricas"]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Rentabilidad (test)", comun.pct(m["rentabilidad"]))
            k2.metric("Sharpe (test)", comun.num(m["sharpe"]))
            k3.metric("Volatilidad", comun.pct(m["volatilidad"]))
            k4.metric("Caída máxima", comun.pct(m["max_dd"]))
            comun.grafica(graficas.curvas_cartera(c["curva"], c["curvas_individuales"], oscuro))
            st.dataframe((c["pesos"] * 100).round(1).rename("Peso %"), width="stretch")
            st.caption("Correlaciones (entrenamiento): cuanto más bajas, más diversifica la cartera.")
            st.dataframe(c["correlaciones"].round(2), width="stretch")
    elif len(elegidas) < 2:
        st.caption("Elige al menos dos estrategias.")
