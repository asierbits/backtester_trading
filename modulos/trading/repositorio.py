"""Acceso a la base de datos específico del módulo de trading (escrituras masivas y consultas)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from core import db
from modulos.trading.backtest.metricas import CLAVES_METRICAS

COLUMNAS_RESULTADOS = [
    "rentabilidad",
    "cagr",
    "volatilidad",
    "sharpe",
    "sortino",
    "max_dd",
    "calmar",
    "n_ops",
    "pct_ganadoras",
    "profit_factor",
    "exposicion",
]


def _limpio(x: Any) -> Any:
    """NaN/inf → None para SQLite/JSON."""
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


# ------------------------------------------------------------------ escrituras
def insertar_estrategias(experimento_id: int, filas: list[tuple[str, dict, str, str, str]]) -> list[int]:
    """Inserta (familia, parámetros, hash, activo, temporalidad) y devuelve los ids en el mismo orden."""
    with db.conexion() as con:
        con.executemany(
            "INSERT INTO estrategias (experimento_id, familia, parametros_json, hash, activo, "
            "temporalidad) VALUES (?, ?, ?, ?, ?, ?)",
            [(experimento_id, f, db.a_json(p), h, a, t) for f, p, h, a, t in filas],
        )
        ids = [
            r[0]
            for r in con.execute("SELECT id FROM estrategias WHERE experimento_id=? ORDER BY id", (experimento_id,))
        ]
    return ids


def insertar_resultados(
    filas: list[tuple[int, str, np.ndarray | dict[str, float]]],
    brutas: dict[tuple[int, str], float] | None = None,
) -> None:
    """Inserta métricas por (estrategia, tramo). Acepta el array del motor o un diccionario."""
    brutas = brutas or {}
    registros = []
    for est_id, tramo, valores in filas:
        m = valores if isinstance(valores, dict) else dict(zip(CLAVES_METRICAS, valores.tolist()))
        m = {k: _limpio(v) for k, v in m.items()}
        registros.append(
            (
                est_id,
                tramo,
                *[m.get(c) for c in COLUMNAS_RESULTADOS],
                _limpio(brutas.get((est_id, tramo))),
                db.a_json(m),
            )
        )
    columnas = ", ".join(COLUMNAS_RESULTADOS)
    marcas = ", ".join(["?"] * (len(COLUMNAS_RESULTADOS) + 4))
    with db.conexion() as con:
        con.executemany(
            f"INSERT OR REPLACE INTO resultados (estrategia_id, tramo, {columnas}, "
            f"rentabilidad_bruta, metricas_json) VALUES ({marcas})",
            registros,
        )


def guardar_referencia(experimento_id: int, activo: str, tramo: str, tipo: str, datos: dict) -> None:
    with db.conexion() as con:
        con.execute(
            "INSERT OR REPLACE INTO referencias VALUES (?, ?, ?, ?, ?)",
            (experimento_id, activo, tramo, tipo, db.a_json(datos)),
        )


def guardar_validaciones(estrategia_id: int, validaciones: dict[str, dict]) -> None:
    with db.conexion() as con:
        con.executemany(
            "INSERT OR REPLACE INTO validaciones VALUES (?, ?, ?, ?)",
            [
                (estrategia_id, tipo, int(bool(v.get("supera", False))), db.a_json(v))
                for tipo, v in validaciones.items()
            ],
        )


def guardar_veredictos(filas: list[tuple[int, str, str, list[str]]]) -> None:
    """(estrategia_id, automático, final, motivos)."""
    with db.conexion() as con:
        con.executemany(
            "INSERT OR REPLACE INTO veredictos VALUES (?, ?, ?, ?)",
            [(i, a, f, db.a_json(m)) for i, a, f, m in filas],
        )


def marcar_finalistas(ids_en_orden: list[int]) -> None:
    with db.conexion() as con:
        con.executemany(
            "UPDATE estrategias SET finalista=1, rango_train=? WHERE id=?",
            [(rango + 1, i) for rango, i in enumerate(ids_en_orden)],
        )


def guardar_operaciones(estrategia_id: int, tramo: str, ops: pd.DataFrame) -> None:
    with db.conexion() as con:
        con.execute("DELETE FROM operaciones WHERE estrategia_id=? AND tramo=?", (estrategia_id, tramo))
        con.executemany(
            "INSERT INTO operaciones (estrategia_id, tramo, entrada, salida, precio_entrada, "
            "precio_salida, comision, resultado, resultado_pct, duracion_velas, direccion, abierta) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    estrategia_id,
                    tramo,
                    str(o.entrada),
                    None if pd.isna(o.salida) else str(o.salida),
                    float(o.precio_entrada),
                    float(o.precio_salida),
                    float(o.comision),
                    float(o.resultado),
                    float(o.resultado_pct),
                    int(o.duracion_velas),
                    int(o.direccion),
                    int(o.abierta),
                )
                for o in ops.itertuples()
            ],
        )


def actualizar_veredicto_final(estrategia_id: int, final: str, motivos: list[str]) -> None:
    with db.conexion() as con:
        con.execute(
            "UPDATE veredictos SET final=?, motivos_json=? WHERE estrategia_id=?",
            (final, db.a_json(motivos), estrategia_id),
        )


def guardar_critica(
    estrategia_id: int,
    agente: str,
    version_prompt: str,
    hash_entrada: str,
    modelo: str,
    respuesta: dict,
    tokens_entrada: int,
    tokens_salida: int,
) -> None:
    with db.conexion() as con:
        con.execute("DELETE FROM criticas_agentes WHERE estrategia_id=? AND agente=?", (estrategia_id, agente))
        con.execute(
            "INSERT INTO criticas_agentes (estrategia_id, agente, version_prompt, hash_entrada, "
            "modelo, puntuacion, recomendacion, respuesta_json, tokens_entrada, tokens_salida, creado) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                estrategia_id,
                agente,
                version_prompt,
                hash_entrada,
                modelo,
                respuesta.get("puntuacion"),
                respuesta.get("recomendacion"),
                db.a_json(respuesta),
                tokens_entrada,
                tokens_salida,
                db.ahora(),
            ),
        )


def cache_leer(clave: str) -> dict | None:
    with db.conexion() as con:
        fila = con.execute(
            "SELECT respuesta_json, tokens_entrada, tokens_salida FROM cache_llm WHERE hash=?",
            (clave,),
        ).fetchone()
    if not fila:
        return None
    return {"respuesta": json.loads(fila[0]), "tokens_entrada": fila[1], "tokens_salida": fila[2]}


def cache_guardar(clave: str, respuesta: dict, tokens_entrada: int, tokens_salida: int) -> None:
    with db.conexion() as con:
        con.execute(
            "INSERT OR REPLACE INTO cache_llm VALUES (?, ?, ?, ?, ?)",
            (clave, db.a_json(respuesta), tokens_entrada, tokens_salida, db.ahora()),
        )


# ------------------------------------------------------------------ consultas
def ranking(experimento_id: int) -> pd.DataFrame:
    """Todas las estrategias del experimento con métricas train/test y veredicto."""
    df = db.consulta_df(
        """
        SELECT e.id, e.familia, e.activo, e.temporalidad, e.parametros_json, e.finalista,
               e.rango_train, v.automatico, v.final AS veredicto, v.motivos_json,
               tr.sharpe AS sharpe_train, te.sharpe AS sharpe_test,
               tr.rentabilidad AS rent_train, te.rentabilidad AS rent_test,
               tr.cagr AS cagr_train, te.cagr AS cagr_test,
               tr.max_dd AS dd_train, te.max_dd AS dd_test,
               tr.calmar AS calmar_train, te.calmar AS calmar_test,
               tr.sortino AS sortino_train, te.sortino AS sortino_test,
               tr.n_ops AS ops_train, te.n_ops AS ops_test,
               tr.pct_ganadoras AS ganadoras_train, te.pct_ganadoras AS ganadoras_test,
               tr.profit_factor AS pf_train, te.profit_factor AS pf_test,
               tr.exposicion AS exposicion_train, te.exposicion AS exposicion_test
        FROM estrategias e
        LEFT JOIN resultados tr ON tr.estrategia_id = e.id AND tr.tramo = 'train'
        LEFT JOIN resultados te ON te.estrategia_id = e.id AND te.tramo = 'test'
        LEFT JOIN veredictos v ON v.estrategia_id = e.id
        WHERE e.experimento_id = ?
        ORDER BY tr.sharpe DESC
        """,
        (experimento_id,),
    )
    if len(df):
        df["parametros"] = df["parametros_json"].map(json.loads)
        df["motivos"] = df["motivos_json"].map(lambda x: json.loads(x) if x else [])
    return df


def estrategia(estrategia_id: int) -> dict | None:
    """Toda la información guardada de una estrategia."""
    base = db.consulta_df("SELECT * FROM estrategias WHERE id=?", (estrategia_id,))
    if base.empty:
        return None
    info = base.iloc[0].to_dict()
    info["parametros"] = json.loads(info.pop("parametros_json"))
    res = db.consulta_df(
        "SELECT tramo, metricas_json, rentabilidad_bruta FROM resultados WHERE estrategia_id=?", (estrategia_id,)
    )
    info["resultados"] = {
        r.tramo: {**json.loads(r.metricas_json), "rentabilidad_bruta": r.rentabilidad_bruta} for r in res.itertuples()
    }
    val = db.consulta_df(
        "SELECT tipo, supera, resultado_json FROM validaciones WHERE estrategia_id=?", (estrategia_id,)
    )
    info["validaciones"] = {r.tipo: json.loads(r.resultado_json) for r in val.itertuples()}
    ver = db.consulta_df("SELECT * FROM veredictos WHERE estrategia_id=?", (estrategia_id,))
    if len(ver):
        info["veredicto_automatico"] = ver.iloc[0]["automatico"]
        info["veredicto"] = ver.iloc[0]["final"]
        info["motivos"] = json.loads(ver.iloc[0]["motivos_json"])
    info["criticas"] = criticas(estrategia_id)
    return info


def criticas(estrategia_id: int) -> list[dict]:
    df = db.consulta_df(
        "SELECT agente, modelo, puntuacion, recomendacion, respuesta_json, tokens_entrada, "
        "tokens_salida, creado FROM criticas_agentes WHERE estrategia_id=? ORDER BY id",
        (estrategia_id,),
    )
    salida = []
    for r in df.itertuples():
        d = json.loads(r.respuesta_json)
        d.update({"modelo": r.modelo, "tokens_entrada": r.tokens_entrada, "tokens_salida": r.tokens_salida})
        salida.append(d)
    return salida


def operaciones(estrategia_id: int, tramo: str | None = None) -> pd.DataFrame:
    sql = "SELECT * FROM operaciones WHERE estrategia_id=?"
    params: tuple = (estrategia_id,)
    if tramo:
        sql += " AND tramo=?"
        params += (tramo,)
    return db.consulta_df(sql + " ORDER BY entrada", params)


def referencias(experimento_id: int) -> pd.DataFrame:
    df = db.consulta_df("SELECT * FROM referencias WHERE experimento_id=?", (experimento_id,))
    if len(df):
        df["datos"] = df["metricas_json"].map(json.loads)
    return df


def finalistas(experimento_id: int) -> list[int]:
    df = db.consulta_df(
        "SELECT id FROM estrategias WHERE experimento_id=? AND finalista=1 ORDER BY rango_train",
        (experimento_id,),
    )
    return df["id"].tolist()


def conteo_veredictos(experimento_id: int) -> dict[str, int]:
    df = db.consulta_df(
        "SELECT v.final, COUNT(*) AS n FROM veredictos v JOIN estrategias e ON e.id=v.estrategia_id "
        "WHERE e.experimento_id=? GROUP BY v.final",
        (experimento_id,),
    )
    return dict(zip(df["final"], df["n"].astype(int)))
