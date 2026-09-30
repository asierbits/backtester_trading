"""Configuración de logs: consola + archivo en data/laboratorio.log."""

from __future__ import annotations

import logging
import re

from core.config import ruta_datos

_PATRON_CLAVE = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")


class _FiltroClaves(logging.Filter):
    """Tapa cualquier clave de API que se cuele en un mensaje de log."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _PATRON_CLAVE.sub("sk-ant-***", str(record.msg))
        return True


_configurado = False


def configurar_logging(nivel: int = logging.INFO) -> None:
    """Configura el logging una sola vez por proceso."""
    global _configurado
    if _configurado:
        return
    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    raiz = logging.getLogger()
    raiz.setLevel(nivel)
    for manejador in (
        logging.StreamHandler(),
        logging.FileHandler(ruta_datos() / "laboratorio.log", encoding="utf-8"),
    ):
        manejador.setFormatter(formato)
        manejador.addFilter(_FiltroClaves())
        raiz.addHandler(manejador)
    for ruidoso in ("urllib3", "httpx", "ccxt", "numba"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    _configurado = True


def obtener_logger(nombre: str) -> logging.Logger:
    """Devuelve un logger con la configuración del laboratorio."""
    configurar_logging()
    return logging.getLogger(nombre)
