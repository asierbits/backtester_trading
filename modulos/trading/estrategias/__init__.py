"""Estrategias del laboratorio. Importar este paquete registra todas las familias."""

from modulos.trading.estrategias import (  # noqa: F401
    bollinger,
    combinada,
    cruce_medias,
    macd,
    rsi_reversion,
    ruptura_donchian,
)
from modulos.trading.estrategias.base import FAMILIAS, Estrategia, crear, hash_estrategia

__all__ = ["FAMILIAS", "Estrategia", "crear", "hash_estrategia"]
