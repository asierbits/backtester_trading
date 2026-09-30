"""Informe HTML autocontenido de un experimento.

Incluye resumen, laboratorio de humo, comparación con el azar, ranking de finalistas,
detalle de cada finalista (curvas, walk-forward, Monte Carlo, motivos y críticas de los
agentes) y un aviso claro de que nada de esto es consejo financiero.

PDF: no se genera directamente (haría falta instalar un motor de renderizado pesado).
El HTML trae estilos de impresión: ábrelo en el navegador y usa "Imprimir → Guardar como PDF".
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import numpy as np

from core import db
from core.config import ruta_informes
from modulos.trading import analisis, graficas
from modulos.trading import repositorio as repo
from modulos.trading.backtest.metricas import NOMBRES_METRICAS, decimal, formatear, miles

ESTILO = """
:root { --fondo:#fcfcfb; --texto:#0b0b0b; --texto2:#52514e; --borde:#e6e5e0; --tarjeta:#ffffff; }
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background: var(--fondo);
       color: var(--texto); max-width: 1100px; margin: 0 auto; padding: 24px 16px; line-height: 1.5; }
h1 { font-size: 1.7rem; margin-bottom: 0; } h2 { margin-top: 2.2rem; border-bottom: 1px solid var(--borde); padding-bottom: 4px; }
h3 { margin-bottom: 4px; }
.sub { color: var(--texto2); margin-top: 4px; }
.aviso { background: #fff6e0; border-left: 4px solid #fab219; padding: 10px 14px; margin: 16px 0; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 16px 0; }
.kpi { background: var(--tarjeta); border: 1px solid var(--borde); border-radius: 8px; padding: 12px; }
.kpi .v { font-size: 1.5rem; font-weight: 600; } .kpi .l { color: var(--texto2); font-size: .85rem; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; margin: 8px 0; }
th, td { border-bottom: 1px solid var(--borde); padding: 5px 8px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.etiqueta { display: inline-block; padding: 1px 8px; border-radius: 10px; font-weight: 600; font-size: .8rem; color: #0b0b0b; }
.APROBADA { background: #c9efc9; } .SOSPECHOSA { background: #ffe7b0; } .DESCARTADA { background: #f6c7c7; }
.motivos li { margin: 2px 0; } .critica { border: 1px solid var(--borde); border-radius: 8px; padding: 8px 12px; margin: 8px 0; background: var(--tarjeta); }
.finalista { page-break-inside: avoid; }
@media print { .grafica { page-break-inside: avoid; } body { max-width: none; } }
"""


def _e(texto: object) -> str:
    return html.escape(str(texto))


def _etiqueta(v: str | None) -> str:
    v = v or "—"
    return f'<span class="etiqueta {_e(v)}">{_e(v)}</span>'


def generar_html(experimento_id: int, n_detalle: int = 10) -> str:
    """Construye el informe completo en HTML."""
    exp = db.obtener_experimento(experimento_id)
    if exp is None:
        raise KeyError(experimento_id)
    r = exp["resumen"]
    rk = repo.ranking(experimento_id)
    refs = repo.referencias(experimento_id)
    primera = [True]

    def fig_html(fig) -> str:  # noqa: ANN001
        incluir = True if primera[0] else False
        primera[0] = False
        return '<div class="grafica">' + fig.to_html(full_html=False, include_plotlyjs=incluir) + "</div>"

    conteo = r.get("conteo", {})
    total = max(r.get("n_estrategias", 1), 1)
    partes = [
        "<!doctype html><html lang='es'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Informe experimento {experimento_id}</title><style>{ESTILO}</style></head><body>",
        f"<h1>Informe del experimento #{experimento_id}</h1>",
        f"<p class='sub'>{_e(exp['nombre'])} · creado {_e(exp['creado'])} · semilla {exp['semilla']} · "
        f"generado {datetime.now():%Y-%m-%d %H:%M}</p>",
        "<div class='aviso'><b>Dinero 100 % ficticio.</b> Esto es un laboratorio de aprendizaje: los "
        "resultados pasados no garantizan resultados futuros y nada de este informe es consejo financiero.</div>",
    ]
    if r.get("datos_sinteticos"):
        partes.append(
            "<div class='aviso'><b>Datos SINTÉTICOS.</b> Los precios se generaron con un proceso aleatorio: "
            "cualquier estrategia que 'gane' aquí lo hace por azar o por la forma del proceso, no por una "
            "ventaja real en el mercado.</div>"
        )
    if "error" in r:
        partes.append(f"<div class='aviso'>El experimento terminó con error: {_e(r['error'])}</div></body></html>")
        return "".join(partes)

    partes += [
        "<h2>Laboratorio de humo</h2>",
        f"<p>Se probaron <b>{miles(total)}</b> estrategias ({miles(r['n_combinaciones'])} combinaciones de "
        f"parámetros × {r['n_activos']} activos, de un espacio de {miles(r['espacio_total'])} posibles). "
        f"<b>{decimal(conteo.get('DESCARTADA', 0) / total * 100)} %</b> resultaron humo (descartadas), "
        f"{decimal(conteo.get('SOSPECHOSA', 0) / total * 100)} % quedaron como sospechosas y "
        f"<b>{conteo.get('APROBADA', 0)}</b> sobrevivieron a todas las pruebas.</p>",
        fig_html(graficas.barras_veredictos(conteo, False)),
    ]
    al = r.get("aleatorias", {})
    rc = al.get("reality_check", {})
    mp = r.get("multiples_pruebas", {})
    partes.append(
        "<div class='kpis'>"
        + "".join(
            f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{_e(t)}</div></div>"
            for v, t in [
                (
                    f"{decimal(al.get('frac_reales_sobre_p95_test', 0) * 100)} %",
                    "estrategias que en test superan al p95 de las aleatorias (el azar daría ~5 %)",
                ),
                (formatear("sharpe", al.get("mejor_sharpe_real_train")), "mejor Sharpe real (entrenamiento)"),
                (
                    formatear("sharpe", rc.get("p95_mejor_azar")),
                    f"p95 del mejor de {miles(total)} aleatorias",
                ),
                (str(mp.get("descubrimientos_bh", 0)), "estrategias significativas tras Benjamini-Hochberg"),
            ]
        )
        + "</div>"
    )
    if len(rk) and len(refs):
        al_test = np.concatenate(
            [np.array(d["sharpe"]) for d in refs[(refs["tipo"] == "aleatoria") & (refs["tramo"] == "test")]["datos"]]
        )
        partes.append(
            fig_html(
                graficas.distribucion_sharpe(
                    rk["sharpe_test"].to_numpy(), al_test, False, "Sharpe en test: estrategias probadas vs aleatorias"
                )
            )
        )
        partes.append(fig_html(graficas.dispersion_train_test(rk, False)))

    # Tabla de finalistas
    fin = rk[rk["finalista"] == 1].sort_values("rango_train")
    partes.append(
        "<h2>Finalistas (elegidas solo con el entrenamiento)</h2><table><tr><th>#</th><th>Estrategia</th>"
        "<th>Activo</th><th>Sharpe train</th><th>Sharpe test</th><th>Rent. test</th><th>Caída test</th>"
        "<th>Ops test</th><th>Veredicto</th></tr>"
    )
    for f in fin.itertuples():
        partes.append(
            f"<tr><td>{f.rango_train}</td><td>{_e(f.familia)} {_e(f.parametros)}</td><td>{_e(f.activo)}</td>"
            f"<td>{formatear('sharpe', f.sharpe_train)}</td><td>{formatear('sharpe', f.sharpe_test)}</td>"
            f"<td>{formatear('rentabilidad', f.rent_test)}</td><td>{formatear('max_dd', f.dd_test)}</td>"
            f"<td>{formatear('n_ops', f.ops_test)}</td><td>{_etiqueta(f.veredicto)}</td></tr>"
        )
    partes.append("</table>")

    # Detalle de cada finalista
    for est_id in fin["id"].tolist()[:n_detalle]:
        partes.append(_seccion_finalista(int(est_id), fig_html))

    partes.append(
        "<h2>Cómo leer este informe</h2><ul>"
        "<li><b>APROBADA</b>: superó todas las pruebas. Aun así, puede fallar en el futuro.</li>"
        "<li><b>SOSPECHOSA</b>: pasó algunas pruebas pero no todas; no hay pruebas suficientes de que la ventaja sea real.</li>"
        "<li><b>DESCARTADA</b>: falla fuera de muestra, tiene pocas operaciones, no se distingue del azar, "
        "depende de un pico de parámetros o desaparece con costes realistas.</li></ul>"
        "</body></html>"
    )
    return "".join(partes)


def _seccion_finalista(est_id: int, fig_html) -> str:  # noqa: ANN001
    d = analisis.detalle(est_id)
    info = d.info
    s = [
        f"<div class='finalista'><h3>#{est_id} · {_e(d.nombre)} · {_e(info['activo'])} {_etiqueta(info.get('veredicto'))}</h3>"
    ]
    s.append("<ul class='motivos'>" + "".join(f"<li>{_e(m)}</li>" for m in info.get("motivos", [])) + "</ul>")
    filas = [
        "rentabilidad",
        "cagr",
        "sharpe",
        "sortino",
        "max_dd",
        "calmar",
        "n_ops",
        "pct_ganadoras",
        "profit_factor",
        "exposicion",
    ]
    res = info.get("resultados", {})
    s.append("<table><tr><th>Métrica</th><th>Entrenamiento</th><th>Test</th><th>Test costes x2</th></tr>")
    for k in filas:
        s.append(
            f"<tr><td>{NOMBRES_METRICAS[k]}</td>"
            + "".join(f"<td>{formatear(k, res.get(t, {}).get(k))}</td>" for t in ("train", "test", "test_x2"))
            + "</tr>"
        )
    s.append(
        f"<tr><td>Rentabilidad bruta (sin costes)</td><td>{formatear('rentabilidad', res.get('train', {}).get('rentabilidad_bruta'))}</td>"
        f"<td>{formatear('rentabilidad', res.get('test', {}).get('rentabilidad_bruta'))}</td><td></td></tr></table>"
    )
    s.append(fig_html(graficas.curva_capital(d.resultado.equity, d.buy_hold.equity, False, d.fecha_corte)))
    val = info.get("validaciones", {})
    if "walk_forward" in val:
        s.append(fig_html(graficas.walk_forward(val["walk_forward"], False)))
    if val.get("monte_carlo", {}).get("valido"):
        s.append(fig_html(graficas.abanico_monte_carlo(val["monte_carlo"], False)))
    for c in info.get("criticas", []):
        s.append(
            f"<div class='critica'><b>{_e(c.get('agente'))}</b> · puntuación {c.get('puntuacion')}/10 · "
            f"recomienda <b>{_e(c.get('recomendacion'))}</b><br>{_e(c.get('explicacion'))}"
            + (f"<br><i>Prueba adicional: {_e(c['prueba_adicional'])}</i>" if c.get("prueba_adicional") else "")
            + "</div>"
        )
    s.append("</div>")
    return "".join(s)


def guardar(experimento_id: int, n_detalle: int = 10) -> Path:
    """Genera el informe y lo guarda en la carpeta de informes."""
    ruta = ruta_informes() / f"experimento_{experimento_id}_{datetime.now():%Y%m%d_%H%M}.html"
    ruta.write_text(generar_html(experimento_id, n_detalle), encoding="utf-8")
    return ruta
