"""Obtención de datos de mercado: Binance (ccxt), datos sintéticos y CSV propio.

Los datos se guardan en Parquet: data/mercado/<fuente>/<ACTIVO>_<temporalidad>.parquet
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from core import db
from core.config import cargar_config, ruta_datos
from core.logging_setup import obtener_logger
from modulos.trading.datos.limpieza import FRECUENCIAS, limpiar, normalizar

log = obtener_logger(__name__)

# Callback de progreso: (fracción 0-1, texto)
Progreso = Callable[[float, str], None]

MS_POR_VELA = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def ruta_parquet(activo: str, temporalidad: str, fuente: str) -> Path:
    """Ruta del Parquet de un activo y temporalidad."""
    carpeta = ruta_datos() / "mercado" / fuente
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta / f"{activo.replace('/', '-')}_{temporalidad}.parquet"


def cargar(activo: str, temporalidad: str, fuente: str) -> pd.DataFrame:
    """Carga velas guardadas. Lanza FileNotFoundError si no existen."""
    ruta = ruta_parquet(activo, temporalidad, fuente)
    if not ruta.exists():
        raise FileNotFoundError(f"No hay datos de {activo} {temporalidad} ({fuente}). Descárgalos en la página Datos.")
    return pd.read_parquet(ruta)


def cargar_o_generar(activo: str, temporalidad: str, fuente: str, semilla: int = 42) -> pd.DataFrame:
    """Carga los datos; si la fuente es 'sintetico' y no existen, los genera al vuelo."""
    try:
        return cargar(activo, temporalidad, fuente)
    except FileNotFoundError:
        if fuente != "sintetico":
            raise
        d = cargar_config()["datos"]
        crear_sinteticos(activo, temporalidad, d["desde"], d["hasta"], semilla)
        return cargar(activo, temporalidad, fuente)


def _guardar(df: pd.DataFrame, activo: str, temporalidad: str, fuente: str) -> dict:
    """Limpia, guarda en Parquet y registra en datos_meta. Devuelve el informe de calidad."""
    salto = cargar_config()["datos"]["salto_anomalo"]
    limpio, informe = limpiar(df, temporalidad, salto)
    ruta = ruta_parquet(activo, temporalidad, fuente)
    limpio.to_parquet(ruta)
    db.guardar_meta_datos(activo, temporalidad, fuente, limpio, informe, str(ruta))
    return informe


# ------------------------------------------------------------------ Binance
def descargar_binance(
    activo: str,
    temporalidad: str,
    desde: str = "2020-01-01",
    hasta: str | None = None,
    progreso: Progreso | None = None,
) -> dict:
    """Descarga velas de Binance (datos públicos, sin clave), con paginación y reanudación.

    Si ya existe un Parquet, continúa desde la última vela guardada. Guarda cada
    cierto número de páginas para no perder lo descargado si se corta la conexión.
    """
    import ccxt  # Importación diferida: no hace falta en modo sin conexión

    bolsa = ccxt.binance({"enableRateLimit": True})
    paso = MS_POR_VELA[temporalidad]
    inicio_ms = int(pd.Timestamp(desde, tz="UTC").timestamp() * 1000)
    fin_ms = int((pd.Timestamp(hasta, tz="UTC") if hasta else pd.Timestamp.now(tz="UTC")).timestamp() * 1000)

    ruta = ruta_parquet(activo, temporalidad, "binance")
    previo = pd.read_parquet(ruta) if ruta.exists() else pd.DataFrame()
    if len(previo):
        # Reanudación: se vuelve a pedir la última vela por si estaba incompleta
        inicio_ms = max(inicio_ms, int(previo.index.max().timestamp() * 1000))
        log.info("Reanudando %s %s desde %s", activo, temporalidad, previo.index.max())

    total = max(1, (fin_ms - inicio_ms) // paso)
    filas: list[list] = []
    cursor = inicio_ms
    barra = tqdm(total=total, desc=f"{activo} {temporalidad}", unit="velas")
    intentos = 0
    while cursor < fin_ms:
        try:
            lote = bolsa.fetch_ohlcv(activo, timeframe=temporalidad, since=cursor, limit=1000)
            intentos = 0
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            intentos += 1
            if intentos > 5:
                log.error("Descarga interrumpida en %s: %s", activo, e)
                break
            espera = 2**intentos
            log.warning("Error de red (%s). Reintento en %ss", e, espera)
            time.sleep(espera)
            continue
        if not lote:
            break
        filas.extend(lote)
        nuevo_cursor = lote[-1][0] + paso
        barra.update(len(lote))
        if progreso:
            progreso(min(1.0, (nuevo_cursor - inicio_ms) / (fin_ms - inicio_ms)), f"{activo} {temporalidad}")
        if nuevo_cursor <= cursor:
            break
        cursor = nuevo_cursor
        # Guardado parcial cada ~20 páginas para poder reanudar
        if len(filas) >= 20_000:
            previo = _combinar(previo, filas)
            previo.to_parquet(ruta)
            filas = []
    barra.close()

    datos = _combinar(previo, filas)
    datos = datos[datos.index < pd.Timestamp(fin_ms, unit="ms", tz="UTC")]
    informe = _guardar(datos, activo, temporalidad, "binance")
    log.info("%s %s: %s velas guardadas", activo, temporalidad, informe["velas_finales"])
    return informe


def _combinar(previo: pd.DataFrame, filas: list[list]) -> pd.DataFrame:
    """Une lo ya guardado con lo nuevo, sin duplicados."""
    if not filas:
        return previo
    nuevo = normalizar(pd.DataFrame(filas, columns=["timestamp", "open", "high", "low", "close", "volume"]))
    todo = pd.concat([previo, nuevo]) if len(previo) else nuevo
    return todo[~todo.index.duplicated(keep="last")].sort_index()


# ------------------------------------------------------------------ Sintéticos
def generar_sinteticos(
    activo: str,
    temporalidad: str,
    desde: str = "2020-01-01",
    hasta: str | None = None,
    semilla: int = 42,
) -> pd.DataFrame:
    """Genera velas sintéticas realistas (sin conexión).

    Proceso: rentabilidades con volatilidad agrupada (tipo GARCH(1,1)), colas gruesas
    (t de Student) y regímenes de tendencia alcista/bajista/lateral que cambian al azar.
    No tienen ninguna ventaja explotable real: sirven para probar el flujo, no para
    sacar conclusiones de trading.
    """
    rng = np.random.default_rng(semilla + sum(map(ord, activo)) + sum(map(ord, temporalidad)))
    fin = pd.Timestamp(hasta, tz="UTC") if hasta else pd.Timestamp.now(tz="UTC").floor("D")
    indice = pd.date_range(pd.Timestamp(desde, tz="UTC"), fin, freq=FRECUENCIAS[temporalidad])
    n = len(indice)
    velas_dia = {"1h": 24, "4h": 6, "1d": 1}[temporalidad]

    # Parámetros diarios típicos de una cripto grande, escalados a la temporalidad
    vol_diaria = 0.035
    omega, alfa, beta = 0.05, 0.08, 0.90  # GARCH sobre varianza normalizada (media 1)
    derivas = np.array([0.0015, -0.0015, 0.0])  # alcista, bajista, lateral (por día)
    duracion_media_reg = 120 * velas_dia

    rent = np.empty(n)
    var = 1.0
    regimen = 2
    choques = rng.standard_t(df=4, size=n) / np.sqrt(2.0)  # varianza 1
    for i in range(n):
        if rng.random() < 1 / duracion_media_reg:
            regimen = rng.integers(0, 3)
        sigma = vol_diaria * np.sqrt(var / velas_dia)
        rent[i] = derivas[regimen] / velas_dia + sigma * choques[i]
        var = omega + alfa * choques[i] ** 2 * var + beta * var
        var = min(var, 25.0)

    precio_inicial = 1000.0 + (sum(map(ord, activo)) % 50) * 200
    close = precio_inicial * np.exp(np.cumsum(rent))
    open_ = np.concatenate([[precio_inicial], close[:-1]]) * np.exp(rng.normal(0, 0.0005, n))
    rango = np.abs(rng.normal(0, vol_diaria / np.sqrt(velas_dia) * 0.6, n))
    high = np.maximum(open_, close) * (1 + rango)
    low = np.minimum(open_, close) * (1 - rango)
    volume = rng.lognormal(mean=10, sigma=0.5, size=n) * (1 + 20 * np.abs(rent))

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(indice, name="timestamp"),
    )


def crear_sinteticos(
    activo: str, temporalidad: str, desde: str = "2020-01-01", hasta: str | None = None, semilla: int = 42
) -> dict:
    """Genera y guarda datos sintéticos (fuente 'sintetico')."""
    df = generar_sinteticos(activo, temporalidad, desde, hasta, semilla)
    return _guardar(df, activo, temporalidad, "sintetico")


# ------------------------------------------------------------------ Acciones (yfinance)
def descargar_yfinance(
    ticker: str,
    temporalidad: str,
    desde: str = "2020-01-01",
    hasta: str | None = None,
) -> dict:
    """Descarga acciones/ETF con yfinance (fuente 'yfinance').

    Limitaciones de Yahoo: las velas horarias solo existen para los últimos ~730 días,
    y no hay velas de 4h (se construyen agregando las horarias). Las acciones solo
    cotizan en horario de mercado, así que no hay velas nocturnas ni de fin de semana
    (el informe de calidad las contará como 'faltantes' en temporalidades intradía).
    """
    import yfinance as yf  # Importación diferida

    intervalo = {"1h": "1h", "4h": "1h", "1d": "1d"}[temporalidad]
    inicio = pd.Timestamp(desde, tz="UTC")
    if intervalo == "1h":
        limite = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=729)
        if inicio < limite:
            log.warning("Yahoo solo da velas horarias de los últimos 730 días: se recorta el inicio")
            inicio = limite
    crudo = yf.download(
        ticker,
        start=inicio.strftime("%Y-%m-%d"),
        end=hasta,
        interval=intervalo,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    if crudo is None or crudo.empty:
        raise ValueError(f"Yahoo no devolvió datos para {ticker} {temporalidad}")
    crudo.columns = [str(c).lower() for c in crudo.columns]
    crudo.index.name = "timestamp"
    df = normalizar(crudo)
    if temporalidad == "4h":
        df = (
            df.resample("4h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
    informe = _guardar(df, ticker, temporalidad, "yfinance")
    log.info("%s %s (yfinance): %s velas", ticker, temporalidad, informe["velas_finales"])
    return informe


# ------------------------------------------------------------------ CSV propio
def importar_csv(archivo, activo: str, temporalidad: str) -> dict:
    """Importa un CSV con columnas timestamp, open, high, low, close, volume (fuente 'csv')."""
    df = pd.read_csv(archivo)
    df.columns = [c.strip().lower() for c in df.columns]
    return _guardar(df, activo, temporalidad, "csv")


def descargar_todo(fuente: str, progreso: Progreso | None = None) -> dict[str, dict]:
    """Descarga (o genera) todos los activos y temporalidades de config.yaml."""
    cfg = cargar_config()
    d = cfg["datos"]
    activos = d.get("acciones", []) if fuente == "yfinance" else d["activos"]
    pares = [(a, t) for a in activos for t in d["temporalidades"]]
    informes = {}
    for i, (activo, tf) in enumerate(pares):
        if progreso:
            progreso(i / len(pares), f"{activo} {tf}")
        if fuente == "binance":
            informes[f"{activo} {tf}"] = descargar_binance(activo, tf, d["desde"], d["hasta"])
        elif fuente == "yfinance":
            informes[f"{activo} {tf}"] = descargar_yfinance(activo, tf, d["desde"], d["hasta"])
        else:
            informes[f"{activo} {tf}"] = crear_sinteticos(activo, tf, d["desde"], d["hasta"], cfg["general"]["semilla"])
    if progreso:
        progreso(1.0, "Terminado")
    return informes


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Descarga los datos definidos en config.yaml")
    p.add_argument("--fuente", choices=["binance", "sintetico", "yfinance"], default="binance")
    args = p.parse_args()
    for clave, inf in descargar_todo(args.fuente).items():
        print(f"{clave}: {inf['velas_finales']} velas, {inf['velas_faltantes']} faltantes")
