"""Paper trading: seguir estrategias APROBADAS sobre precios actuales con dinero ficticio.

Cómo funciona cada actualización:
1. Descarga las velas recientes (Binance, sin clave) con suficiente historia previa para
   calcular los indicadores.
2. Descarta la vela en curso (aún no ha cerrado): solo se decide con velas cerradas.
3. Recalcula las señales y simula con el MISMO motor que el backtest desde la fecha de
   alta. Así lo simulado en vivo es comparable con lo que predijo el backtest.
4. Registra las operaciones nuevas y, si hay alertas configuradas, avisa.

Nunca se envían órdenes a ningún broker: todo es simulado.

Bucle continuo:  python -m modulos.trading.paper.paper_trading --bucle
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from core import db
from core.config import cargar_config
from core.logging_setup import obtener_logger
from modulos.trading import repositorio as repo
from modulos.trading.backtest.costes import Costes
from modulos.trading.backtest.motor import Mercado, aplicar_sizing_volatilidad, backtest
from modulos.trading.datos.descarga import MS_POR_VELA, cargar_o_generar
from modulos.trading.datos.limpieza import normalizar
from modulos.trading.estrategias import crear
from modulos.trading.paper import alertas

log = obtener_logger(__name__)

VELAS_CALENTAMIENTO = 1000  # Historia previa para los indicadores (medias de hasta 250, filtros de 200…)


def registrar(
    estrategia_id: int,
    fuente: str | None = None,
    capital: float | None = None,
    inicio: str | pd.Timestamp | None = None,
) -> int:
    """Da de alta una estrategia en paper trading. Devuelve el id de seguimiento."""
    info = repo.estrategia(estrategia_id)
    if info is None:
        raise KeyError(f"No existe la estrategia {estrategia_id}")
    exp = db.obtener_experimento(int(info["experimento_id"]))
    cfg = exp["config"]
    fuente = fuente or cargar_config().get("paper", {}).get("fuente", "binance")
    capital = capital or float(cfg["backtest"]["capital_inicial"])
    paso = pd.Timedelta(milliseconds=MS_POR_VELA[info["temporalidad"]])
    if inicio is None:
        if fuente == "binance":
            inicio = pd.Timestamp.now(tz="UTC").floor(paso) + paso  # desde la próxima vela
        else:
            # Modo demostración sin conexión: se "reproduce" el último 10 % de los datos guardados
            df = cargar_o_generar(info["activo"], info["temporalidad"], fuente, int(cfg["general"]["semilla"]))
            inicio = df.index[int(len(df) * 0.9)]
    esperado = {
        "test": info["resultados"].get("test", {}),
        "veredicto": info.get("veredicto"),
    }
    nombre = crear(info["familia"], info["parametros"]).descripcion_corta() + f" · {info['activo']}"
    with db.conexion() as con:
        cur = con.execute(
            "INSERT INTO paper_estrategias (estrategia_id, nombre, familia, parametros_json, activo, "
            "temporalidad, fuente, capital, creado, inicio, esperado_json, estado_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                estrategia_id,
                nombre,
                info["familia"],
                db.a_json(info["parametros"]),
                info["activo"],
                info["temporalidad"],
                fuente,
                capital,
                db.ahora(),
                str(pd.Timestamp(inicio)),
                db.a_json(esperado),
                db.a_json({"config_backtest": cfg["backtest"]}),
            ),
        )
        return int(cur.lastrowid)


def listar() -> pd.DataFrame:
    df = db.consulta_df("SELECT * FROM paper_estrategias ORDER BY id DESC")
    if len(df):
        df["estado"] = df["estado_json"].map(lambda x: json.loads(x) if x else {})
        df["esperado"] = df["esperado_json"].map(lambda x: json.loads(x) if x else {})
    return df


def operaciones(paper_id: int) -> pd.DataFrame:
    return db.consulta_df("SELECT * FROM paper_operaciones WHERE paper_id=? ORDER BY entrada", (paper_id,))


def pausar(paper_id: int, activa: bool) -> None:
    with db.conexion() as con:
        con.execute("UPDATE paper_estrategias SET activa=? WHERE id=?", (int(activa), paper_id))


def borrar(paper_id: int) -> None:
    with db.conexion() as con:
        con.execute("DELETE FROM paper_estrategias WHERE id=?", (paper_id,))


# ------------------------------------------------------------------ velas
def velas_binance(activo: str, temporalidad: str, desde: pd.Timestamp) -> pd.DataFrame:
    """Velas CERRADAS desde `desde` (paginando). Requiere conexión."""
    import ccxt

    bolsa = ccxt.binance({"enableRateLimit": True})
    paso = MS_POR_VELA[temporalidad]
    cursor = int(desde.timestamp() * 1000)
    ahora_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    filas: list[list] = []
    while cursor < ahora_ms:
        lote = bolsa.fetch_ohlcv(activo, timeframe=temporalidad, since=cursor, limit=1000)
        if not lote:
            break
        filas.extend(lote)
        nuevo = lote[-1][0] + paso
        if nuevo <= cursor:
            break
        cursor = nuevo
    df = normalizar(pd.DataFrame(filas, columns=["timestamp", "open", "high", "low", "close", "volume"]))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    cierre = df.index + pd.Timedelta(milliseconds=paso)
    return df[cierre <= pd.Timestamp.now(tz="UTC")]  # fuera la vela en curso


def obtener_velas(fila: pd.Series) -> pd.DataFrame:
    paso = pd.Timedelta(milliseconds=MS_POR_VELA[fila["temporalidad"]])
    inicio = pd.Timestamp(fila["inicio"])
    desde = inicio - VELAS_CALENTAMIENTO * paso
    if fila["fuente"] == "binance":
        try:
            return velas_binance(fila["activo"], fila["temporalidad"], desde)
        except Exception as e:  # noqa: BLE001
            raise ConnectionError(
                f"No se pudo conectar con Binance ({type(e).__name__}). Revisa tu conexión a internet; "
                "sin conexión puedes seguir la estrategia en modo demostración con la fuente del experimento."
            ) from e
    df = cargar_o_generar(fila["activo"], fila["temporalidad"], fila["fuente"])
    return df[df.index >= desde]


# ------------------------------------------------------------------ actualización
def actualizar(paper_id: int, enviar_alertas: bool | None = None) -> dict:
    """Actualiza una estrategia en paper trading. Devuelve su estado."""
    tabla = listar()
    fila = tabla[tabla["id"] == paper_id]
    if fila.empty:
        raise KeyError(f"No existe el seguimiento {paper_id}")
    fila = fila.iloc[0]
    cfg_bt = fila["estado"].get("config_backtest") or cargar_config()["backtest"]
    if enviar_alertas is None:
        enviar_alertas = bool(cargar_config().get("paper", {}).get("alertas", False))

    df = obtener_velas(fila)
    inicio = pd.Timestamp(fila["inicio"])
    ini = int(np.searchsorted(df.index, inicio))
    if ini >= len(df):
        estado = {**fila["estado"], "mensaje": "Aún no hay velas cerradas desde el alta", "config_backtest": cfg_bt}
        _guardar_estado(paper_id, estado)
        return estado

    mercado = Mercado(df, fila["temporalidad"], cfg_bt.get("precio_ejecucion", "apertura"))
    estrategia = crear(fila["familia"], json.loads(fila["parametros_json"]))
    pos = estrategia.posiciones(df, cortos=bool(cfg_bt.get("permitir_cortos")))
    if cfg_bt.get("sizing") == "volatilidad":
        pos = aplicar_sizing_volatilidad(pos, mercado.close, mercado.velas_ano, float(cfg_bt["volatilidad_objetivo"]))
    costes = Costes.desde_config(cfg_bt)
    res = backtest(mercado, pos, costes, float(fila["capital"]), ini, len(df))

    nuevas = _guardar_operaciones(paper_id, res.operaciones)
    abiertas = res.operaciones[res.operaciones["abierta"]]
    posicion_actual = int(abiertas["direccion"].iloc[-1]) if len(abiertas) else 0
    senal = int(np.sign(pos[-1]))
    ultima = str(df.index[-1])
    estado = {
        "config_backtest": cfg_bt,
        "ultima_vela": ultima,
        "velas_seguidas": int(len(df) - ini),
        "equity": float(res.equity.iloc[-1]),
        "rentabilidad": float(res.equity.iloc[-1] / float(fila["capital"]) - 1),
        "posicion": posicion_actual,
        "senal_siguiente_vela": senal,
        "metricas": res.metricas,
        "curva": {
            "fechas": [str(x) for x in res.equity.index[:: max(1, len(res.equity) // 300)]],
            "equity": [float(x) for x in res.equity.to_numpy()[:: max(1, len(res.equity) // 300)]],
        },
        "ultima_senal_alertada": fila["estado"].get("ultima_senal_alertada"),
    }
    textos = []
    if senal != posicion_actual and estado["ultima_senal_alertada"] != ultima:
        accion = {1: "COMPRA (largo)", -1: "VENTA EN CORTO", 0: "CERRAR posición"}[senal]
        textos.append(
            f"[Paper] {fila['nombre']}: señal de {accion} al cierre de {ultima}. "
            f"Se ejecutaría en la próxima vela. Precio de cierre: {df['close'].iloc[-1]:.4f}. "
            "Dinero ficticio: no es una recomendación de inversión."
        )
        estado["ultima_senal_alertada"] = ultima
    for o in nuevas:
        textos.append(
            f"[Paper] {fila['nombre']}: operación simulada {'cerrada' if not o['abierta'] else 'abierta'} "
            f"({o['entrada']} → {o['salida'] or 'abierta'}), resultado {o['resultado_pct'] * 100:.2f} %."
        )
    estado["alertas"] = textos
    if enviar_alertas and textos:
        estado["alertas_enviadas_por"] = alertas.enviar("\n".join(textos))
        with db.conexion() as con:
            con.execute("UPDATE paper_operaciones SET alertada=1 WHERE paper_id=?", (paper_id,))
    _guardar_estado(paper_id, estado)
    return estado


def _guardar_estado(paper_id: int, estado: dict) -> None:
    with db.conexion() as con:
        con.execute(
            "UPDATE paper_estrategias SET estado_json=?, ultima_actualizacion=? WHERE id=?",
            (db.a_json(estado), db.ahora(), paper_id),
        )


def _guardar_operaciones(paper_id: int, ops: pd.DataFrame) -> list[dict]:
    """Inserta o actualiza operaciones. Devuelve las nuevas o las que se acaban de cerrar."""
    cambios = []
    with db.conexion() as con:
        previas = {
            r[0]: bool(r[1])
            for r in con.execute("SELECT entrada, abierta FROM paper_operaciones WHERE paper_id=?", (paper_id,))
        }
        for o in ops.itertuples():
            entrada = str(o.entrada)
            salida = None if pd.isna(o.salida) else str(o.salida)
            registro = {
                "entrada": entrada,
                "salida": salida,
                "abierta": bool(o.abierta),
                "resultado_pct": float(o.resultado_pct),
            }
            if entrada not in previas or (previas[entrada] and not o.abierta):
                cambios.append(registro)
            con.execute(
                "INSERT INTO paper_operaciones (paper_id, entrada, salida, precio_entrada, precio_salida, "
                "direccion, comision, resultado, resultado_pct, abierta) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(paper_id, entrada) DO UPDATE SET salida=excluded.salida, "
                "precio_salida=excluded.precio_salida, comision=excluded.comision, "
                "resultado=excluded.resultado, resultado_pct=excluded.resultado_pct, abierta=excluded.abierta",
                (
                    paper_id,
                    entrada,
                    salida,
                    float(o.precio_entrada),
                    float(o.precio_salida),
                    int(o.direccion),
                    float(o.comision),
                    float(o.resultado),
                    float(o.resultado_pct),
                    int(o.abierta),
                ),
            )
    return cambios


def actualizar_todas(enviar_alertas: bool | None = None) -> dict[int, dict]:
    """Actualiza todas las estrategias activas; los fallos de una no paran las demás."""
    salida = {}
    tabla = listar()
    for fila in tabla[tabla["activa"] == 1].itertuples() if len(tabla) else []:
        try:
            salida[fila.id] = actualizar(int(fila.id), enviar_alertas)
        except Exception as e:  # noqa: BLE001
            log.exception("Fallo al actualizar el paper %s", fila.id)
            salida[fila.id] = {"error": str(e)}
    return salida


def comparar_con_backtest(paper_id: int) -> pd.DataFrame:
    """Tabla: lo que predijo el backtest (test) frente a lo simulado en vivo."""
    fila = listar().set_index("id").loc[paper_id]
    esperado = fila["esperado"].get("test", {})
    real = fila["estado"].get("metricas", {})
    claves = ["cagr", "sharpe", "max_dd", "pct_ganadoras", "profit_factor", "exposicion"]
    return pd.DataFrame(
        {"Backtest (test)": [esperado.get(k) for k in claves], "Paper (en vivo)": [real.get(k) for k in claves]},
        index=claves,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Actualiza el paper trading")
    p.add_argument("--bucle", action="store_true", help="Repetir cada 'intervalo_minutos' de config.yaml")
    args = p.parse_args()
    while True:
        for pid, est in actualizar_todas().items():
            print(
                pid,
                est.get("error") or f"equity {est.get('equity', 0):.2f} € · señal {est.get('senal_siguiente_vela')}",
            )
        if not args.bucle:
            break
        time.sleep(60 * int(cargar_config().get("paper", {}).get("intervalo_minutos", 60)))
