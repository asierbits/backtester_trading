"""Base de datos SQLite: esquema y funciones de acceso.

Todo lo que produce un experimento se guarda aquí (configuración exacta, semilla,
estrategias, métricas, operaciones, validaciones, críticas y veredictos), para que
cualquier resultado sea reproducible y auditable.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from core.config import ruta_datos

ESQUEMA = """
CREATE TABLE IF NOT EXISTS experimentos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL,
    creado TEXT NOT NULL,
    config_json TEXT NOT NULL,
    semilla INTEGER NOT NULL,
    n_combinaciones INTEGER DEFAULT 0,
    n_estrategias INTEGER DEFAULT 0,
    estado TEXT DEFAULT 'en_curso',
    duracion_s REAL,
    resumen_json TEXT
);

CREATE TABLE IF NOT EXISTS estrategias (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experimento_id INTEGER NOT NULL REFERENCES experimentos(id) ON DELETE CASCADE,
    familia TEXT NOT NULL,
    parametros_json TEXT NOT NULL,
    hash TEXT NOT NULL,
    activo TEXT NOT NULL,
    temporalidad TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_estrategias_exp ON estrategias(experimento_id);

CREATE TABLE IF NOT EXISTS resultados (
    estrategia_id INTEGER NOT NULL REFERENCES estrategias(id) ON DELETE CASCADE,
    tramo TEXT NOT NULL,
    rentabilidad REAL, cagr REAL, volatilidad REAL, sharpe REAL, sortino REAL,
    max_dd REAL, calmar REAL, n_ops INTEGER, pct_ganadoras REAL, profit_factor REAL,
    exposicion REAL, rentabilidad_bruta REAL,
    metricas_json TEXT,
    PRIMARY KEY (estrategia_id, tramo)
);

CREATE TABLE IF NOT EXISTS operaciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    estrategia_id INTEGER NOT NULL REFERENCES estrategias(id) ON DELETE CASCADE,
    tramo TEXT NOT NULL,
    entrada TEXT, salida TEXT,
    precio_entrada REAL, precio_salida REAL,
    comision REAL, resultado REAL, resultado_pct REAL, duracion_velas INTEGER
);
CREATE INDEX IF NOT EXISTS ix_operaciones_est ON operaciones(estrategia_id);

CREATE TABLE IF NOT EXISTS referencias (
    experimento_id INTEGER NOT NULL REFERENCES experimentos(id) ON DELETE CASCADE,
    activo TEXT NOT NULL,
    tramo TEXT NOT NULL,
    tipo TEXT NOT NULL,          -- buy_hold | aleatoria
    metricas_json TEXT NOT NULL, -- para 'aleatoria': distribución de Sharpe y rentabilidad
    PRIMARY KEY (experimento_id, activo, tramo, tipo)
);

CREATE TABLE IF NOT EXISTS validaciones (
    estrategia_id INTEGER NOT NULL REFERENCES estrategias(id) ON DELETE CASCADE,
    tipo TEXT NOT NULL,
    supera INTEGER,
    resultado_json TEXT,
    PRIMARY KEY (estrategia_id, tipo)
);

CREATE TABLE IF NOT EXISTS veredictos (
    estrategia_id INTEGER PRIMARY KEY REFERENCES estrategias(id) ON DELETE CASCADE,
    automatico TEXT NOT NULL,
    final TEXT NOT NULL,
    motivos_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS criticas_agentes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    estrategia_id INTEGER NOT NULL REFERENCES estrategias(id) ON DELETE CASCADE,
    agente TEXT NOT NULL,
    version_prompt TEXT NOT NULL,
    hash_entrada TEXT NOT NULL,
    modelo TEXT,
    puntuacion REAL,
    recomendacion TEXT,
    respuesta_json TEXT,
    tokens_entrada INTEGER, tokens_salida INTEGER,
    creado TEXT
);

CREATE TABLE IF NOT EXISTS cache_llm (
    hash TEXT PRIMARY KEY,
    respuesta_json TEXT NOT NULL,
    tokens_entrada INTEGER, tokens_salida INTEGER,
    creado TEXT
);

CREATE TABLE IF NOT EXISTS datos_meta (
    activo TEXT NOT NULL,
    temporalidad TEXT NOT NULL,
    fuente TEXT NOT NULL,
    desde TEXT, hasta TEXT,
    n_velas INTEGER,
    calidad_json TEXT,
    actualizado TEXT,
    ruta TEXT,
    PRIMARY KEY (activo, temporalidad, fuente)
);
"""


def ruta_db() -> Path:
    """Ruta del archivo SQLite."""
    return ruta_datos() / "laboratorio.sqlite"


def ahora() -> str:
    """Fecha y hora actual en UTC, formato ISO."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def conexion(ruta: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Abre una conexión con el esquema creado y hace commit al salir."""
    con = sqlite3.connect(ruta or ruta_db(), timeout=30)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    con.executescript(ESQUEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def a_json(obj: Any) -> str:
    """Serializa a JSON con claves ordenadas (útil para hashes estables)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=_por_defecto)


def _por_defecto(obj: Any) -> Any:
    """Convierte tipos de numpy/pandas a tipos nativos para JSON."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"No serializable: {type(obj)}")


def consulta_df(sql: str, parametros: tuple = ()) -> pd.DataFrame:
    """Ejecuta una consulta y devuelve un DataFrame."""
    with conexion() as con:
        return pd.read_sql_query(sql, con, params=parametros)


# ---------------------------------------------------------------- experimentos
def crear_experimento(nombre: str, config: dict, semilla: int) -> int:
    """Registra un experimento nuevo y devuelve su id."""
    with conexion() as con:
        cur = con.execute(
            "INSERT INTO experimentos (nombre, creado, config_json, semilla) VALUES (?, ?, ?, ?)",
            (nombre, ahora(), a_json(config), semilla),
        )
        return int(cur.lastrowid)


def cerrar_experimento(
    experimento_id: int, estado: str, duracion_s: float, resumen: dict, n_comb: int, n_est: int
) -> None:
    """Marca el experimento como terminado (o fallido) y guarda su resumen."""
    with conexion() as con:
        con.execute(
            "UPDATE experimentos SET estado=?, duracion_s=?, resumen_json=?, "
            "n_combinaciones=?, n_estrategias=? WHERE id=?",
            (estado, duracion_s, a_json(resumen), n_comb, n_est, experimento_id),
        )


def listar_experimentos() -> pd.DataFrame:
    """Todos los experimentos, del más reciente al más antiguo."""
    return consulta_df(
        "SELECT id, nombre, creado, estado, n_combinaciones, n_estrategias, duracion_s, semilla, "
        "resumen_json FROM experimentos ORDER BY id DESC"
    )


def obtener_experimento(experimento_id: int) -> dict | None:
    """Un experimento con su configuración decodificada."""
    df = consulta_df("SELECT * FROM experimentos WHERE id=?", (experimento_id,))
    if df.empty:
        return None
    fila = df.iloc[0].to_dict()
    fila["config"] = json.loads(fila.pop("config_json"))
    fila["resumen"] = json.loads(fila["resumen_json"]) if fila["resumen_json"] else {}
    return fila


def borrar_experimento(experimento_id: int) -> None:
    """Elimina un experimento y todo lo que cuelga de él."""
    with conexion() as con:
        con.execute("DELETE FROM experimentos WHERE id=?", (experimento_id,))


# ---------------------------------------------------------------- datos_meta
def guardar_meta_datos(
    activo: str, temporalidad: str, fuente: str, df: pd.DataFrame, calidad: dict, ruta: str
) -> None:
    """Registra qué datos hay descargados y hasta qué fecha."""
    with conexion() as con:
        con.execute(
            "INSERT OR REPLACE INTO datos_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                activo,
                temporalidad,
                fuente,
                str(df.index.min()) if len(df) else None,
                str(df.index.max()) if len(df) else None,
                len(df),
                a_json(calidad),
                ahora(),
                ruta,
            ),
        )


def listar_meta_datos() -> pd.DataFrame:
    """Inventario de datos disponibles."""
    return consulta_df("SELECT * FROM datos_meta ORDER BY fuente, activo, temporalidad")
