"""Gráficas Plotly compartidas por las páginas y los informes.

Paleta: categórica en orden fijo (azul, naranja, aqua…), escala divergente azul↔rojo con
gris neutro en el centro para valores con signo (Sharpe), y colores de estado reservados
para los veredictos (siempre acompañados de su etiqueta de texto). Tema claro y oscuro
con superficies y tintas propias, no una inversión automática.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modulos.trading.backtest.metricas import decimal, miles

CATEGORICA = {
    "claro": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "oscuro": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
SUPERFICIE = {"claro": "#fcfcfb", "oscuro": "#1a1a19"}
TEXTO = {"claro": "#0b0b0b", "oscuro": "#ffffff"}
TEXTO_2 = {"claro": "#52514e", "oscuro": "#c3c2b7"}
REJILLA = {"claro": "#e6e5e0", "oscuro": "#33332f"}
NEUTRO = {"claro": "#8a8984", "oscuro": "#8f8e87"}
DIVERGENTE = {
    "claro": [[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]],
    "oscuro": [[0, "#e66767"], [0.5, "#383835"], [1, "#3987e5"]],
}
ESTADO = {"APROBADA": "#0ca30c", "SOSPECHOSA": "#fab219", "DESCARTADA": "#d03b3b"}


def modo(oscuro: bool) -> str:
    return "oscuro" if oscuro else "claro"


def estilo(fig: go.Figure, oscuro: bool, titulo: str | None = None, alto: int = 380) -> go.Figure:
    """Aplica el tema del laboratorio (ejes y rejilla discretos)."""
    m = modo(oscuro)
    fig.update_layout(
        template="plotly_dark" if oscuro else "plotly_white",
        title={
            "text": titulo,
            "font": {"size": 15, "color": TEXTO[m]},
            "x": 0.01,
            "xanchor": "left",
            "y": 0.98,
            "yref": "container",
            "yanchor": "top",
        }
        if titulo
        else None,
        paper_bgcolor=SUPERFICIE[m],
        plot_bgcolor=SUPERFICIE[m],
        font={"color": TEXTO_2[m], "size": 12},
        colorway=CATEGORICA[m],
        height=alto,
        margin={"l": 50, "r": 20, "t": 80 if titulo else 40, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        hovermode="x unified",
        hoverlabel={"bgcolor": SUPERFICIE[m], "font": {"color": TEXTO[m]}},
    )
    fig.update_xaxes(gridcolor=REJILLA[m], zeroline=False, linecolor=REJILLA[m])
    fig.update_yaxes(gridcolor=REJILLA[m], zeroline=False, linecolor=REJILLA[m])
    return fig


def _marca_corte(fig: go.Figure, fecha: Any, oscuro: bool, **kw: Any) -> None:
    if fecha is None:
        return
    fig.add_vline(x=fecha, line_dash="dot", line_width=1, line_color=NEUTRO[modo(oscuro)], **kw)
    fig.add_annotation(
        x=fecha,
        y=1,
        yref="paper",
        text="inicio del test",
        showarrow=False,
        xanchor="left",
        font={"size": 11, "color": TEXTO_2[modo(oscuro)]},
        **kw,
    )


def curva_capital(
    equity: pd.Series,
    bh: pd.Series | None,
    oscuro: bool,
    corte: Any = None,
    titulo: str = "Curva de capital vs comprar y mantener",
) -> go.Figure:
    m = modo(oscuro)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity,
            name="Estrategia",
            line={"width": 2, "color": CATEGORICA[m][0]},
            hovertemplate="%{y:,.2f} €<extra>Estrategia</extra>",
        )
    )
    if bh is not None:
        fig.add_trace(
            go.Scatter(
                x=bh.index,
                y=bh,
                name="Comprar y mantener",
                line={"width": 2, "color": CATEGORICA[m][1]},
                hovertemplate="%{y:,.2f} €<extra>Comprar y mantener</extra>",
            )
        )
    _marca_corte(fig, corte, oscuro)
    fig.update_yaxes(title="Capital (€ ficticios)")
    return estilo(fig, oscuro, titulo)


def caida(equity: pd.Series, oscuro: bool, corte: Any = None) -> go.Figure:
    dd = equity / equity.cummax() - 1
    m = modo(oscuro)
    fig = go.Figure(
        go.Scatter(
            x=dd.index,
            y=dd * 100,
            fill="tozeroy",
            name="Caída",
            line={"width": 1.5, "color": CATEGORICA[m][7]},
            hovertemplate="%{y:.1f} %<extra>Caída desde máximos</extra>",
        )
    )
    _marca_corte(fig, corte, oscuro)
    fig.update_yaxes(title="Caída desde máximos (%)")
    return estilo(fig, oscuro, "Caída desde máximos (drawdown)", alto=260)


def precio_con_operaciones(df: pd.DataFrame, ops: pd.DataFrame, oscuro: bool, max_velas: int = 3000) -> go.Figure:
    """Velas (o línea si hay demasiadas) con marcas de entrada y salida."""
    m = modo(oscuro)
    datos = df.iloc[-max_velas:] if len(df) > max_velas else df
    fig = go.Figure()
    if len(datos) <= 1500:
        fig.add_trace(
            go.Candlestick(
                x=datos.index,
                open=datos["open"],
                high=datos["high"],
                low=datos["low"],
                close=datos["close"],
                name="Precio",
                increasing_line_color=CATEGORICA[m][0],
                decreasing_line_color=CATEGORICA[m][7],
            )
        )
    else:
        fig.add_trace(
            go.Scatter(x=datos.index, y=datos["close"], name="Cierre", line={"width": 1.2, "color": NEUTRO[m]})
        )
    if len(ops):
        o = ops.copy()
        o["entrada"] = pd.to_datetime(o["entrada"], utc=True)
        o["salida"] = pd.to_datetime(o["salida"], utc=True)
        o = o[o["entrada"] >= datos.index[0]]
        largos = o[o["direccion"] > 0]
        cortos = o[o["direccion"] < 0]
        for sub, simbolo, nombre in (
            (largos, "triangle-up", "Entrada largo"),
            (cortos, "triangle-down", "Entrada corto"),
        ):
            if len(sub):
                fig.add_trace(
                    go.Scatter(
                        x=sub["entrada"],
                        y=sub["precio_entrada"],
                        mode="markers",
                        name=nombre,
                        marker={
                            "symbol": simbolo,
                            "size": 10,
                            "color": CATEGORICA[m][2],
                            "line": {"width": 1, "color": SUPERFICIE[m]},
                        },
                        hovertemplate="%{x}<br>%{y:,.4f}<extra>" + nombre + "</extra>",
                    )
                )
        cerradas = o[o["salida"].notna()]
        if len(cerradas):
            fig.add_trace(
                go.Scatter(
                    x=cerradas["salida"],
                    y=cerradas["precio_salida"],
                    mode="markers",
                    name="Salida",
                    marker={"symbol": "x", "size": 9, "color": CATEGORICA[m][1]},
                    customdata=np.stack([cerradas["resultado_pct"] * 100], axis=-1),
                    hovertemplate="%{x}<br>%{y:,.4f}<br>resultado %{customdata[0]:.2f} %<extra>Salida</extra>",
                )
            )
    fig.update_layout(xaxis_rangeslider_visible=False)
    titulo = "Precio y operaciones" + (f" (últimas {max_velas} velas)" if len(df) > max_velas else "")
    return estilo(fig, oscuro, titulo, alto=420)


def distribucion_operaciones(ops: pd.DataFrame, oscuro: bool) -> go.Figure:
    m = modo(oscuro)
    r = ops["resultado_pct"] * 100
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=r[r > 0],
            name="Ganadoras",
            marker_color=CATEGORICA[m][0],
            nbinsx=30,
            marker_line={"width": 1, "color": SUPERFICIE[m]},
        )
    )
    fig.add_trace(
        go.Histogram(
            x=r[r <= 0],
            name="Perdedoras",
            marker_color=CATEGORICA[m][7],
            nbinsx=30,
            marker_line={"width": 1, "color": SUPERFICIE[m]},
        )
    )
    fig.update_layout(barmode="overlay", hovermode="closest")
    fig.update_xaxes(title="Resultado por operación (%)")
    fig.update_yaxes(title="Nº de operaciones")
    return estilo(fig, oscuro, "Distribución de resultados por operación", alto=300)


def mapa_calor(tabla: pd.DataFrame, oscuro: bool, marcar: tuple[Any, Any] | None = None) -> go.Figure:
    """Sharpe en función de dos parámetros (escala divergente centrada en 0)."""
    m = modo(oscuro)
    valores = tabla.to_numpy(dtype=float)
    limite = float(np.nanmax(np.abs(valores))) if np.isfinite(valores).any() else 1.0
    fig = go.Figure(
        go.Heatmap(
            z=valores,
            x=[str(c) for c in tabla.columns],
            y=[str(i) for i in tabla.index],
            colorscale=DIVERGENTE[m],
            zmid=0,
            zmin=-limite,
            zmax=limite,
            xgap=2,
            ygap=2,
            colorbar={"title": "Sharpe"},
            hovertemplate=f"{tabla.columns.name}=%{{x}}<br>{tabla.index.name}=%{{y}}<br>Sharpe %{{z:.2f}}<extra></extra>",
        )
    )
    if marcar is not None:
        fig.add_trace(
            go.Scatter(
                x=[str(marcar[0])],
                y=[str(marcar[1])],
                mode="markers",
                name="Estrategia elegida",
                marker={"symbol": "square-open", "size": 24, "color": TEXTO[m], "line": {"width": 3}},
                hoverinfo="skip",
            )
        )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title=tabla.columns.name, type="category")
    fig.update_yaxes(title=tabla.index.name, type="category")
    return estilo(fig, oscuro, "Mapa de calor de parámetros (Sharpe en entrenamiento)", alto=460)


def walk_forward(wf: dict, oscuro: bool) -> go.Figure:
    m = modo(oscuro)
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=("Curva encadenada fuera de muestra", "Rentabilidad por ventana"),
    )
    curva = wf.get("curva", {})
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(curva.get("fechas", [])),
            y=curva.get("equity", []),
            name="Walk-forward (re-optimizando)",
            line={"width": 2, "color": CATEGORICA[m][0]},
        ),
        row=1,
        col=1,
    )
    etiquetas = [f"{v['desde'][:7]}" for v in wf["ventanas"]]
    fig.add_trace(
        go.Bar(
            x=etiquetas,
            y=[v["rentabilidad_oos"] * 100 for v in wf["ventanas"]],
            name="Re-optimizando",
            marker_color=CATEGORICA[m][0],
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=etiquetas,
            y=[v["rentabilidad_fijos"] * 100 for v in wf["ventanas"]],
            name="Parámetros fijos",
            marker_color=CATEGORICA[m][1],
        ),
        row=1,
        col=2,
    )
    if all(v.get("rentabilidad_bh") is not None for v in wf["ventanas"]):
        fig.add_trace(
            go.Bar(
                x=etiquetas,
                y=[v["rentabilidad_bh"] * 100 for v in wf["ventanas"]],
                name="Comprar y mantener",
                marker_color=CATEGORICA[m][2],
            ),
            row=1,
            col=2,
        )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, hovermode="closest")
    fig.update_yaxes(title_text="€", row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=2)
    return estilo(fig, oscuro, None, alto=360)


def abanico_monte_carlo(mc: dict, oscuro: bool) -> go.Figure:
    m = modo(oscuro)
    color = CATEGORICA[m][0]
    ab = mc["abanico"]
    x = list(range(1, len(ab["50"]) + 1))
    fig = go.Figure()
    for tray in mc.get("trayectorias", [])[:30]:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=tray,
                mode="lines",
                line={"width": 0.6, "color": NEUTRO[m]},
                opacity=0.35,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(go.Scatter(x=x, y=ab["95"], line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=x, y=ab["5"], fill="tonexty", fillcolor=_alfa(color, 0.18), line={"width": 0}, name="Percentiles 5–95"
        )
    )
    fig.add_trace(go.Scatter(x=x, y=ab["75"], line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=x, y=ab["25"], fill="tonexty", fillcolor=_alfa(color, 0.35), line={"width": 0}, name="Percentiles 25–75"
        )
    )
    fig.add_trace(go.Scatter(x=x, y=ab["50"], line={"width": 2, "color": color}, name="Mediana"))
    fig.update_xaxes(title="Nº de operación")
    fig.update_yaxes(title="Capital (€ ficticios)")
    fig.update_layout(hovermode="x")
    return estilo(fig, oscuro, f"Monte Carlo: {miles(mc['n_sim'])} remuestreos de las operaciones de test", alto=380)


def distribucion_sharpe(reales: np.ndarray, aleatorias: np.ndarray, oscuro: bool, titulo: str) -> go.Figure:
    """Histograma comparado: estrategias reales vs aleatorias (densidad)."""
    m = modo(oscuro)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=reales,
            name="Estrategias probadas",
            histnorm="probability density",
            marker_color=CATEGORICA[m][0],
            opacity=0.75,
            nbinsx=60,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=aleatorias,
            name="Estrategias aleatorias",
            histnorm="probability density",
            marker_color=CATEGORICA[m][1],
            opacity=0.6,
            nbinsx=40,
        )
    )
    fig.update_layout(barmode="overlay", hovermode="closest")
    fig.update_xaxes(title="Sharpe")
    fig.update_yaxes(title="Densidad")
    return estilo(fig, oscuro, titulo, alto=340)


def dispersion_train_test(df: pd.DataFrame, oscuro: bool) -> go.Figure:
    """Sharpe train (x) vs test (y) coloreado por veredicto (con leyenda)."""
    fig = go.Figure()
    muestra = df.sample(min(len(df), 6000), random_state=0) if len(df) > 6000 else df
    for v in ("DESCARTADA", "SOSPECHOSA", "APROBADA"):
        sub = muestra[muestra["veredicto"] == v]
        if len(sub):
            fig.add_trace(
                go.Scattergl(
                    x=sub["sharpe_train"],
                    y=sub["sharpe_test"],
                    mode="markers",
                    name=v.capitalize(),
                    marker={
                        "size": 8 if v != "DESCARTADA" else 5,
                        "color": ESTADO[v],
                        "opacity": 0.85 if v != "DESCARTADA" else 0.35,
                    },
                    customdata=sub[["id", "familia", "activo"]].to_numpy(),
                    hovertemplate="#%{customdata[0]} %{customdata[1]} %{customdata[2]}<br>train %{x:.2f} · test %{y:.2f}<extra></extra>",
                )
            )
    lim = [float(np.nanmin(muestra["sharpe_train"])), float(np.nanmax(muestra["sharpe_train"]))]
    fig.add_trace(
        go.Scatter(
            x=lim,
            y=lim,
            mode="lines",
            name="test = train",
            line={"dash": "dot", "width": 1, "color": NEUTRO[modo(oscuro)]},
        )
    )
    fig.update_xaxes(title="Sharpe en entrenamiento")
    fig.update_yaxes(title="Sharpe en test")
    fig.update_layout(hovermode="closest")
    return estilo(fig, oscuro, "Lo que prometía el pasado vs lo que pasó después", alto=440)


def barras_veredictos(conteo: dict[str, int], oscuro: bool) -> go.Figure:
    orden = ["DESCARTADA", "SOSPECHOSA", "APROBADA"]
    valores = [conteo.get(v, 0) for v in orden]
    total = max(sum(valores), 1)
    fig = go.Figure(
        go.Bar(
            y=[v.capitalize() for v in orden],
            x=valores,
            orientation="h",
            marker_color=[ESTADO[v] for v in orden],
            text=[f"{miles(n)} ({decimal(n / total * 100)} %)" for n in valores],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:,}<extra></extra>",
        )
    )
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(title="Nº de estrategias", range=[0, max(valores) * 1.25 + 1])
    return estilo(fig, oscuro, None, alto=220)


def curvas_cartera(curva: pd.Series, individuales: pd.DataFrame, oscuro: bool) -> go.Figure:
    m = modo(oscuro)
    fig = go.Figure()
    for k, col in enumerate(individuales.columns[:7]):
        fig.add_trace(
            go.Scatter(
                x=individuales.index,
                y=individuales[col],
                name=col,
                line={"width": 1.2, "color": CATEGORICA[m][(k + 1) % 8]},
                opacity=0.7,
            )
        )
    fig.add_trace(go.Scatter(x=curva.index, y=curva, name="Cartera", line={"width": 3, "color": CATEGORICA[m][0]}))
    fig.update_yaxes(title="Crecimiento de 1 €")
    return estilo(fig, oscuro, "Cartera combinada en el tramo de test", alto=400)


def _alfa(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"
