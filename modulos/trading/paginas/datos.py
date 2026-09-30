"""Datos: descargar/actualizar activos, ver calidad, importar CSV y gráfica de velas."""

import json

import streamlit as st

from core.config import cargar_config
from modulos.trading import graficas
from modulos.trading.datos import descarga
from modulos.trading.paginas import comun

st.title("Datos de mercado")
cfg = cargar_config()
d = cfg["datos"]

pestanas = st.tabs(["Descargar", "Calidad", "Importar CSV", "Gráfica"])

with pestanas[0]:
    st.write(
        "Los datos se guardan en Parquet en `data/mercado/<fuente>/`. La descarga de Binance usa "
        "datos públicos (sin clave), respeta los límites de la API y **reanuda** donde se quedó."
    )
    fuente = st.radio(
        "Fuente",
        ["binance", "sintetico", "yfinance"],
        format_func={
            "binance": "Binance (cripto, real)",
            "sintetico": "Datos de ejemplo sintéticos (sin conexión)",
            "yfinance": "Yahoo Finance (acciones/ETF)",
        }.get,
        horizontal=True,
    )
    lista = d.get("acciones", []) if fuente == "yfinance" else d["activos"]
    c1, c2, c3 = st.columns([2, 1, 1])
    activos = c1.multiselect("Activos", lista, default=lista)
    temporalidades = c2.multiselect("Temporalidades", ["1h", "4h", "1d"], default=d["temporalidades"])
    desde = c3.text_input("Desde", d["desde"])
    if fuente == "sintetico":
        st.info(
            "Los datos sintéticos imitan la forma de una cripto (volatilidad agrupada, colas gruesas, "
            "regímenes) pero **no contienen ninguna ventaja real**: sirven para probar la app."
        )
    if fuente == "yfinance":
        st.caption(
            "Yahoo solo ofrece velas horarias de los últimos 730 días; las de 4h se construyen agregando las horarias."
        )
    if st.button("Descargar / actualizar", type="primary", icon=":material/download:"):
        barra = st.progress(0.0, "Empezando…")
        pares = [(a, t) for a in activos for t in temporalidades]
        errores = []
        for i, (a, t) in enumerate(pares):
            barra.progress(i / max(len(pares), 1), f"{a} {t}")
            try:
                if fuente == "binance":
                    inf = descarga.descargar_binance(
                        a,
                        t,
                        desde,
                        d["hasta"],
                        progreso=lambda f, txt, i=i: barra.progress((i + f) / len(pares), txt),
                    )
                elif fuente == "yfinance":
                    inf = descarga.descargar_yfinance(a, t, desde, d["hasta"])
                else:
                    inf = descarga.crear_sinteticos(a, t, desde, d["hasta"], cfg["general"]["semilla"])
                st.write(
                    f"✓ {a} {t}: {comun.miles(inf['velas_finales'])} velas · "
                    f"{comun.miles(inf['velas_faltantes'])} faltantes"
                )
            except Exception as e:  # noqa: BLE001
                errores.append(f"{a} {t}: {e}")
        barra.progress(1.0, "Terminado")
        for err in errores:
            st.error(err)
        if errores and fuente == "binance":
            st.info("¿Sin conexión? Usa los datos de ejemplo sintéticos para probar el laboratorio.")

with pestanas[1]:
    meta = comun.tabla_estado_datos()
    if meta.empty:
        st.info("No hay datos todavía.")
    else:
        st.dataframe(meta, hide_index=True, width="stretch")
        st.caption(
            "Nunca se rellenan huecos con datos inventados: las velas faltantes se cuentan y se "
            "reportan. Las velas imposibles (precio ≤ 0, máximo < mínimo) y los duplicados se eliminan."
        )
        from core import db

        crudo = db.listar_meta_datos()
        elegido = st.selectbox(
            "Informe completo de calidad",
            crudo.index,
            format_func=lambda i: f"{crudo.loc[i, 'fuente']} · {crudo.loc[i, 'activo']} {crudo.loc[i, 'temporalidad']}",
        )
        st.json(json.loads(crudo.loc[elegido, "calidad_json"]))

with pestanas[2]:
    st.write(
        "Sube un CSV con columnas `timestamp, open, high, low, close, volume` (timestamp en fecha o en segundos/milisegundos)."
    )
    archivo = st.file_uploader("Archivo CSV", type=["csv"])
    c1, c2 = st.columns(2)
    nombre = c1.text_input("Nombre del activo", "MI/ACTIVO")
    tf = c2.selectbox("Temporalidad", ["1h", "4h", "1d"], index=2)
    if archivo and st.button("Importar", type="primary"):
        try:
            inf = descarga.importar_csv(archivo, nombre, tf)
            st.success(f"Importado: {inf['velas_finales']} velas (fuente 'csv').")
            st.json(inf)
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo importar: {e}")

with pestanas[3]:
    from core import db

    crudo = db.listar_meta_datos()
    if crudo.empty:
        st.info("No hay datos todavía.")
    else:
        i = st.selectbox(
            "Serie",
            crudo.index,
            key="serie_grafica",
            format_func=lambda i: f"{crudo.loc[i, 'fuente']} · {crudo.loc[i, 'activo']} {crudo.loc[i, 'temporalidad']}",
        )
        fila = crudo.loc[i]
        df = descarga.cargar(fila["activo"], fila["temporalidad"], fila["fuente"])
        n = st.slider("Velas a mostrar", 100, min(3000, len(df)), min(500, len(df)), step=50)
        import pandas as pd

        comun.grafica(graficas.precio_con_operaciones(df.iloc[-n:], pd.DataFrame(), comun.oscuro(), max_velas=n))
