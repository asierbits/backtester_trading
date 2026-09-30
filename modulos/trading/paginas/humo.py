"""Laboratorio de humo: cuántas estrategias eran humo y cómo se comparan con el azar."""

import numpy as np
import pandas as pd
import streamlit as st

from modulos.trading import graficas
from modulos.trading import repositorio as repo
from modulos.trading.estrategias import FAMILIAS
from modulos.trading.paginas import comun

st.title("Laboratorio de humo")
exp = comun.selector_experimento()
if exp is None:
    st.stop()
r = exp["resumen"]
comun.aviso_sinteticos(r)
oscuro = comun.oscuro()
rk = repo.ranking(int(exp["id"]))
refs = repo.referencias(int(exp["id"]))
conteo = repo.conteo_veredictos(int(exp["id"]))
total = max(sum(conteo.values()), 1)
humo = conteo.get("DESCARTADA", 0)
aprobadas = conteo.get("APROBADA", 0)

st.markdown(
    f"### Probé **{comun.miles(total)}** estrategias: el **{comun.decimal(humo / total * 100)} %** eran humo y "
    f"**{aprobadas}** sobrevivieron."
)
st.write(
    f"Otras {comun.miles(conteo.get('SOSPECHOSA', 0))} quedaron como sospechosas: pasaron algunas pruebas, pero no "
    "hay pruebas suficientes de que su ventaja sea real."
)
comun.grafica(graficas.barras_veredictos(conteo, oscuro))

al = r.get("aleatorias", {})
rc = al.get("reality_check", {})
mp = r.get("multiples_pruebas", {})
st.subheader("¿Mejor que tirar una moneda?")
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Superan al p95 de las aleatorias en test",
    f"{comun.decimal(al.get('frac_reales_sobre_p95_test', 0) * 100)} %",
    "por azar saldría ~5 %",
    delta_color="off",
    delta_arrow="off",
)
c2.metric(
    "Sharpe mediano en test",
    comun.num(al.get("mediana_real_test")),
    f"aleatorias: {comun.num(al.get('mediana_aleatoria_test'))}",
    delta_color="off",
    delta_arrow="off",
)
c3.metric(
    "Mejor Sharpe real (entrenamiento)",
    comun.num(al.get("mejor_sharpe_real_train")),
    f"p95 del mejor por azar: {comun.num(rc.get('p95_mejor_azar'))}",
    delta_color="off",
    delta_arrow="off",
)
c4.metric(
    "Significativas tras Benjamini-Hochberg",
    f"{mp.get('descubrimientos_bh', 0)}",
    f"Bonferroni: {mp.get('bonferroni', 0)}",
    delta_color="off",
    delta_arrow="off",
)

frac = al.get("frac_reales_sobre_p95_test", 0)
if frac <= 0.08:
    st.info(
        f"Solo el {comun.decimal(frac * 100)} % de las estrategias supera en test al percentil 95 de las aleatorias. "
        "Si todo fuera azar, esperaríamos alrededor de un 5 %: el conjunto de estrategias apenas se distingue de la suerte.",
        icon=":material/casino:",
    )
else:
    st.info(
        f"El {comun.decimal(frac * 100)} % supera al percentil 95 de las aleatorias en test (el azar daría ~5 %). Hay algo "
        "más que suerte en parte del conjunto, pero eso no significa que cada estrategia concreta sea fiable.",
        icon=":material/insights:",
    )
st.caption(
    f"Tras probar {comun.miles(total)} estrategias, el Sharpe máximo que cabría esperar por pura suerte es ≈ "
    f"{comun.num(mp.get('sharpe_max_esperado_azar'))} (anualizado, fórmula del Deflated Sharpe Ratio). "
    "Cualquier 'campeona' por debajo de esa cifra es compatible con el azar."
)

if len(rk) and len(refs):
    al_test = np.concatenate(
        [np.array(d["sharpe"]) for d in refs[(refs["tipo"] == "aleatoria") & (refs["tramo"] == "test")]["datos"]]
    )
    al_train = np.concatenate(
        [np.array(d["sharpe"]) for d in refs[(refs["tipo"] == "aleatoria") & (refs["tramo"] == "train")]["datos"]]
    )
    c1, c2 = st.columns(2)
    with c1:
        comun.grafica(
            graficas.distribucion_sharpe(
                rk["sharpe_train"].to_numpy(), al_train, oscuro, "Entrenamiento: probadas vs aleatorias"
            )
        )
    with c2:
        comun.grafica(
            graficas.distribucion_sharpe(rk["sharpe_test"].to_numpy(), al_test, oscuro, "Test: probadas vs aleatorias")
        )
    comun.grafica(graficas.dispersion_train_test(rk, oscuro))
    corr = float(pd.Series(rk["sharpe_train"]).corr(pd.Series(rk["sharpe_test"])))
    st.caption(
        f"Correlación entre el Sharpe de entrenamiento y el de test: {corr:.2f}. Cerca de 0 = lo que "
        "funcionó en el pasado no dice nada de lo que pasa después (la señal típica del humo).".replace(".", ",", 1)
    )

st.subheader("Por familia")
if len(rk):
    tabla = (
        rk.groupby("familia")
        .agg(
            estrategias=("id", "size"),
            descartadas=("veredicto", lambda v: (v == "DESCARTADA").mean() * 100),
            sospechosas=("veredicto", lambda v: (v == "SOSPECHOSA").mean() * 100),
            aprobadas=("veredicto", lambda v: int((v == "APROBADA").sum())),
            sharpe_train_mediano=("sharpe_train", "median"),
            sharpe_test_mediano=("sharpe_test", "median"),
        )
        .reset_index()
    )
    tabla["familia"] = tabla["familia"].map(lambda f: FAMILIAS[f].nombre_legible if f in FAMILIAS else f)
    st.dataframe(
        tabla.round(2),
        hide_index=True,
        width="stretch",
        column_config={
            "descartadas": st.column_config.NumberColumn("% descartadas", format="%.1f"),
            "sospechosas": st.column_config.NumberColumn("% sospechosas", format="%.1f"),
        },
    )

st.subheader("Comprar y mantener en el mismo periodo")
filas = []
for activo, d in r.get("activos", {}).items():
    filas.append(
        {
            "Activo": activo,
            "Rent. train %": d["bh_train"]["rentabilidad"] * 100,
            "Sharpe train": d["bh_train"]["sharpe"],
            "Rent. test %": d["bh_test"]["rentabilidad"] * 100,
            "Sharpe test": d["bh_test"]["sharpe"],
            "Caída test %": d["bh_test"]["max_dd"] * 100,
        }
    )
st.dataframe(pd.DataFrame(filas).round(2), hide_index=True, width="stretch")
comun.aviso_ficticio()
