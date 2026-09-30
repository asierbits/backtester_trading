"""Reglas que deciden APROBADA / SOSPECHOSA / DESCARTADA, con motivos legibles.

- **DESCARTADA**: falla fuera de muestra, tiene pocas operaciones, no se distingue del
  azar, depende de un pico de parámetros o desaparece con costes más altos.
- **APROBADA**: supera TODAS las pruebas (train/test con degradación aceptable,
  walk-forward, robustez, Monte Carlo, múltiples pruebas, azar, buy & hold ajustado
  por riesgo, doble coste y regímenes).
- **SOSPECHOSA**: todo lo demás (cumple algunas pero no todas).

La estadística manda: los agentes de IA solo pueden endurecer el veredicto (ver `endurecer`).
"""

from __future__ import annotations

from typing import Any

APROBADA, SOSPECHOSA, DESCARTADA = "APROBADA", "SOSPECHOSA", "DESCARTADA"
GRAVEDAD = {APROBADA: 0, SOSPECHOSA: 1, DESCARTADA: 2}
DESDE_RECOMENDACION = {"aprobar": APROBADA, "dudar": SOSPECHOSA, "descartar": DESCARTADA}
COLORES = {APROBADA: "#0ca30c", SOSPECHOSA: "#fab219", DESCARTADA: "#d03b3b"}


def num(x: float | None, dec: int = 2) -> str:
    """Número con coma decimal."""
    if x is None:
        return "—"
    return f"{x:.{dec}f}".replace(".", ",")


def pct(x: float | None, dec: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.{dec}f} %".replace(".", ",")


def veredicto_basico(
    train: dict[str, float],
    test: dict[str, float],
    p_aleatoria_train: float,
    cfg: dict[str, Any],
) -> tuple[str, list[str]]:
    """Veredicto para las estrategias que no pasan a la validación completa."""
    v = cfg["veredicto"]
    motivos: list[str] = []
    if train["n_ops"] < v["min_operaciones"]:
        motivos.append(
            f"✗ Solo {int(train['n_ops'])} operaciones en entrenamiento (mínimo "
            f"{v['min_operaciones']}): sin significancia estadística."
        )
    if test["n_ops"] < v["min_operaciones_test"]:
        motivos.append(f"✗ Solo {int(test['n_ops'])} operaciones en test (mínimo {v['min_operaciones_test']}).")
    if train["sharpe"] <= p_aleatoria_train:
        motivos.append(
            f"✗ Su Sharpe de entrenamiento ({num(train['sharpe'])}) no supera al percentil "
            f"{v['percentil_aleatorias']} de las estrategias aleatorias ({num(p_aleatoria_train)}): "
            "compatible con la pura suerte."
        )
    if test["sharpe"] <= 0:
        motivos.append(
            f"✗ Falla fuera de muestra: el Sharpe pasa de {num(train['sharpe'])} en entrenamiento "
            f"a {num(test['sharpe'])} en test."
        )
    if motivos:
        return DESCARTADA, motivos
    motivos.append(f"✓ Sharpe {num(train['sharpe'])} en entrenamiento y {num(test['sharpe'])} en test.")
    if v.get("exigir_validacion_completa", True):
        motivos.append(
            "• No estaba entre las finalistas, así que no pasó walk-forward, Monte Carlo, "
            "robustez ni corrección por múltiples pruebas: como mucho puede ser SOSPECHOSA."
        )
        return SOSPECHOSA, motivos
    return SOSPECHOSA, motivos


def veredicto_completo(val: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> tuple[str, list[str]]:
    """Veredicto de una finalista a partir de todas sus validaciones."""
    v = cfg["veredicto"]
    fallos_graves: list[str] = []
    fallos: list[str] = []
    aciertos: list[str] = []

    ops = val["min_operaciones"]
    if not ops["supera"]:
        fallos_graves.append(
            f"✗ Pocas operaciones: {ops['train']} en entrenamiento y {ops['test']} en test "
            f"(mínimos {v['min_operaciones']} y {v['min_operaciones_test']})."
        )
    else:
        aciertos.append(f"✓ Operaciones suficientes ({ops['train']} train / {ops['test']} test).")

    tt = val["train_test"]
    texto_tt = f"el Sharpe pasa de {num(tt['sharpe_train'])} en entrenamiento a {num(tt['sharpe_test'])} en test"
    if tt["sharpe_test"] <= 0:
        fallos_graves.append(f"✗ Falla fuera de muestra: {texto_tt}.")
    elif not tt["supera"]:
        fallos.append(
            f"✗ Degradación excesiva: {texto_tt} (conserva {pct(tt['conservacion'], 0)}; "
            f"mínimo {pct(v['max_degradacion_sharpe'], 0)})."
        )
    else:
        aciertos.append(f"✓ Aguanta fuera de muestra: {texto_tt}.")

    al = val["aleatoria"]
    if al["supera"]:
        aciertos.append(
            f"✓ En test supera al percentil {v['percentil_aleatorias']} de las aleatorias "
            f"(Sharpe {num(al['sharpe_test'])} vs {num(al['umbral_test'])})."
        )
    else:
        fallos.append(
            f"✗ En test no se distingue del azar: Sharpe {num(al['sharpe_test'])} frente al "
            f"percentil {v['percentil_aleatorias']} de las aleatorias ({num(al['umbral_test'])})."
        )

    bh = val["buy_hold"]
    if bh["supera"]:
        aciertos.append(
            f"✓ Mejor que comprar y mantener ajustado por riesgo en test (Sharpe {num(bh['sharpe'])} "
            f"vs {num(bh['sharpe_bh'])}; Calmar {num(bh['calmar'])} vs {num(bh['calmar_bh'])})."
        )
    else:
        fallos.append(
            f"✗ No supera a comprar y mantener en test ni en Sharpe ({num(bh['sharpe'])} vs "
            f"{num(bh['sharpe_bh'])}) ni en Calmar ({num(bh['calmar'])} vs {num(bh['calmar_bh'])})."
        )

    dc = val["doble_coste"]
    if not dc["supera"]:
        fallos_graves.append(
            f"✗ Desaparece con costes x{num(dc['multiplicador'], 0)}: en test el Sharpe cae a "
            f"{num(dc['sharpe'])} y la rentabilidad a {pct(dc['rentabilidad'])}."
        )
    else:
        aciertos.append(f"✓ Resiste costes x{num(dc['multiplicador'], 0)} (Sharpe test {num(dc['sharpe'])}).")

    rb = val.get("robustez", {})
    if rb.get("valido"):
        texto = (
            f"el {pct(rb['frac_positivas'], 0)} de {rb['n_vecinos']} vecinos sigue con Sharpe > 0 y "
            f"su mediana conserva el {pct(rb['ratio_mediana'], 0)} del original"
        )
        if rb["supera"]:
            aciertos.append(f"✓ Robusta a cambios de parámetros: {texto}.")
        else:
            fallos_graves.append(f"✗ Depende de un pico de parámetros: {texto}.")

    wf = val.get("walk_forward", {})
    if wf:
        texto = (
            f"{pct(wf['pct_ventanas_rentables'], 0)} de ventanas fuera de muestra rentables, "
            f"rentabilidad encadenada {pct(wf['rentabilidad_oos_total'])}"
        )
        (aciertos if wf["supera"] else fallos).append(
            ("✓ Walk-forward: " if wf["supera"] else "✗ Walk-forward flojo: ") + texto + "."
        )

    mc = val.get("monte_carlo", {})
    if mc.get("valido"):
        texto = (
            f"intervalo 5–95 % de rentabilidad [{pct(mc['rentabilidad_p5'])}, "
            f"{pct(mc['rentabilidad_p95'])}], caída máxima p95 {pct(mc['max_dd_p95'])}, "
            f"probabilidad de perder {pct(mc['prob_perdida'], 0)}"
        )
        (aciertos if mc["supera"] else fallos).append(
            ("✓ Monte Carlo: " if mc["supera"] else "✗ Monte Carlo arriesgado: ") + texto + "."
        )

    mp = val.get("multiples_pruebas", {})
    if mp:
        (aciertos if mp["supera"] else fallos).append(("✓ " if mp["supera"] else "✗ ") + mp["explicacion"])

    rg = val.get("regimenes", {})
    if rg:
        if rg["supera"]:
            aciertos.append("✓ No depende solo del mercado alcista.")
        elif rg.get("solo_gana_en_alcista"):
            fallos.append("✗ Solo gana cuando el mercado sube: pierde en mercados bajistas y laterales.")
        else:
            fallos.append(f"✗ Pierde más que el mercado en régimen {', '.join(rg['fallos'])}.")

    if fallos_graves:
        return DESCARTADA, fallos_graves + fallos + aciertos
    if fallos:
        return SOSPECHOSA, fallos + aciertos
    return APROBADA, aciertos


def endurecer(automatico: str, recomendacion_agentes: str | None) -> str:
    """Los agentes solo pueden endurecer el veredicto, nunca relajarlo."""
    if not recomendacion_agentes:
        return automatico
    propuesto = DESDE_RECOMENDACION.get(recomendacion_agentes, automatico)
    return propuesto if GRAVEDAD[propuesto] > GRAVEDAD[automatico] else automatico


def resumen_texto(veredicto: str, motivos: list[str]) -> str:
    """Frase corta: 'Descartada: el Sharpe cae de ...'."""
    primero = next((m for m in motivos if m.startswith("✗")), motivos[0] if motivos else "")
    return f"{veredicto.capitalize()}: {primero.lstrip('✗✓• ').rstrip('.')}"
