"""Cliente de la API de Claude: reintentos con espera exponencial, caché, control de coste
y validación del JSON devuelto.

- La clave se lee de .env (ANTHROPIC_API_KEY) y nunca se registra.
- Respuestas en JSON con esquema fijo (salidas estructuradas de la API) y, además,
  validación local: si algo no cuadra, se reintenta y se registra el error.
- Caché en SQLite: misma entrada + misma versión de prompt + mismo modelo = no se repite la llamada.
- Presupuesto por experimento: límite de llamadas y de tokens.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.config import clave_anthropic
from core.logging_setup import obtener_logger
from modulos.trading import repositorio as repo
from modulos.trading.agentes.prompts import VERSION_PROMPTS

log = obtener_logger(__name__)

CARACTERES_POR_TOKEN = 3.2  # Estimación prudente para texto en español con JSON


class ErrorAgente(Exception):
    """Fallo no recuperable de una llamada (petición inválida, rechazo, JSON inválido tras reintentos)."""


class PresupuestoAgotado(ErrorAgente):
    """Se alcanzó el límite de llamadas o de tokens del experimento."""


@dataclass
class Uso:
    """Consumo acumulado."""

    llamadas: int = 0
    desde_cache: int = 0
    tokens_entrada: int = 0
    tokens_salida: int = 0
    errores: int = 0

    def coste(self, precio_entrada: float, precio_salida: float) -> float:
        return (self.tokens_entrada * precio_entrada + self.tokens_salida * precio_salida) / 1e6


def validar_critica(d: Any, arbitro: bool = False) -> list[str]:
    """Errores de formato de una crítica (lista vacía = válida)."""
    errores: list[str] = []
    if not isinstance(d, dict):
        return ["La respuesta no es un objeto JSON"]
    for clave in ("agente", "explicacion", "recomendacion"):
        if not isinstance(d.get(clave), str) or not d.get(clave).strip():
            errores.append(f"'{clave}' debe ser un texto no vacío")
    p = d.get("puntuacion")
    if not isinstance(p, (int, float)) or isinstance(p, bool) or not 0 <= p <= 10:
        errores.append("'puntuacion' debe ser un número entre 0 y 10")
    if d.get("recomendacion") not in ("aprobar", "dudar", "descartar"):
        errores.append("'recomendacion' debe ser aprobar, dudar o descartar")
    for clave in ("preocupaciones", "puntos_fuertes"):
        v = d.get(clave)
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            errores.append(f"'{clave}' debe ser una lista de textos")
    if arbitro:
        if d.get("veredicto_final") not in ("APROBADA", "SOSPECHOSA", "DESCARTADA"):
            errores.append("'veredicto_final' debe ser APROBADA, SOSPECHOSA o DESCARTADA")
        if not isinstance(d.get("prueba_adicional"), str):
            errores.append("'prueba_adicional' debe ser un texto")
    return errores


def estimar_tokens(texto: str) -> int:
    return int(len(texto) / CARACTERES_POR_TOKEN) + 1


class ClienteIA:
    """Envoltorio de la API con presupuesto, caché y reintentos. Seguro entre hilos."""

    def __init__(self, cfg_agentes: dict[str, Any], api: Any | None = None) -> None:
        self.cfg = cfg_agentes
        self.modelo = cfg_agentes["modelo"]
        self._api = api
        self.uso = Uso()
        self._cerrojo = threading.Lock()

    # ------------------------------------------------------------------ utilidades
    @property
    def api(self) -> Any:
        if self._api is None:
            import anthropic

            clave = clave_anthropic()
            if not clave:
                raise ErrorAgente("Falta ANTHROPIC_API_KEY en el archivo .env")
            # Reintentos propios (con espera exponencial y registro), no los del SDK
            self._api = anthropic.Anthropic(api_key=clave, max_retries=0, timeout=600.0)
        return self._api

    def clave_cache(self, sistema: str, usuario: str, esquema: dict) -> str:
        texto = "|".join(
            [
                self.modelo,
                VERSION_PROMPTS,
                str(self.cfg.get("esfuerzo")),
                sistema,
                usuario,
                json.dumps(esquema, sort_keys=True),
            ]
        )
        return hashlib.sha256(texto.encode()).hexdigest()

    def coste_actual(self) -> float:
        return self.uso.coste(self.cfg["precio_entrada_millon"], self.cfg["precio_salida_millon"])

    def estimar_coste(self, entradas: list[str]) -> dict[str, float]:
        """Estimación previa (sin llamar a la API) para una lista de mensajes de entrada."""
        t_in = sum(estimar_tokens(t) for t in entradas)
        t_out = len(entradas) * int(self.cfg.get("tokens_salida_estimados", 2500))
        coste = (t_in * self.cfg["precio_entrada_millon"] + t_out * self.cfg["precio_salida_millon"]) / 1e6
        return {"llamadas": len(entradas), "tokens_entrada": t_in, "tokens_salida": t_out, "coste_usd": coste}

    def _reservar(self) -> None:
        with self._cerrojo:
            if self.uso.llamadas >= int(self.cfg["max_llamadas_por_experimento"]):
                raise PresupuestoAgotado(f"Límite de {self.cfg['max_llamadas_por_experimento']} llamadas alcanzado")
            if self.uso.tokens_entrada + self.uso.tokens_salida >= int(self.cfg["max_tokens_por_experimento"]):
                raise PresupuestoAgotado(f"Límite de {self.cfg['max_tokens_por_experimento']:,} tokens alcanzado")
            self.uso.llamadas += 1

    def _llamar_api(self, sistema: str, usuario: str, esquema: dict) -> Any:
        params: dict[str, Any] = {
            "model": self.modelo,
            "max_tokens": int(self.cfg.get("max_tokens", 8000)),
            "system": sistema,
            "messages": [{"role": "user", "content": usuario}],
            "output_config": {
                "effort": self.cfg.get("esfuerzo", "medium"),
                "format": {"type": "json_schema", "schema": esquema},
            },
        }
        if self.cfg.get("usar_fallbacks", True):
            # Si el modelo rechaza la petición, la API la repite con el modelo de respaldo recomendado
            return self.api.beta.messages.create(
                **params, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
            )
        return self.api.messages.create(**params)

    # ------------------------------------------------------------------ llamada principal
    def pedir_json(
        self,
        sistema: str,
        usuario: str,
        esquema: dict,
        validador: Callable[[Any], list[str]],
    ) -> tuple[dict, dict]:
        """Devuelve (respuesta_json, info) con info = tokens, caché, hash."""
        import anthropic

        clave = self.clave_cache(sistema, usuario, esquema)
        guardado = repo.cache_leer(clave)
        if guardado and not validador(guardado["respuesta"]):
            with self._cerrojo:
                self.uso.desde_cache += 1
            return guardado["respuesta"], {**guardado, "cache": True, "hash": clave}

        max_reintentos = int(self.cfg.get("max_reintentos", 4))
        ultimo_error = "desconocido"
        for intento in range(max_reintentos + 1):
            if intento:
                espera = min(60.0, 2**intento) + random.uniform(0, 1)
                log.warning("Reintento %s/%s en %.1f s (%s)", intento, max_reintentos, espera, ultimo_error)
                time.sleep(espera)
            self._reservar()
            try:
                resp = self._llamar_api(sistema, usuario, esquema)
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
                ultimo_error = f"{type(e).__name__}"
                self.uso.errores += 1
                continue
            except anthropic.APIStatusError as e:
                if e.status_code >= 500 or e.status_code == 429:
                    ultimo_error = f"HTTP {e.status_code}"
                    self.uso.errores += 1
                    continue
                self.uso.errores += 1
                raise ErrorAgente(f"La API rechazó la petición (HTTP {e.status_code}): {e.message}") from e

            uso = getattr(resp, "usage", None)
            t_in = int(getattr(uso, "input_tokens", 0) or 0)
            t_out = int(getattr(uso, "output_tokens", 0) or 0)
            with self._cerrojo:
                self.uso.tokens_entrada += t_in
                self.uso.tokens_salida += t_out

            if resp.stop_reason == "refusal":
                self.uso.errores += 1
                raise ErrorAgente("El modelo declinó responder (refusal)")
            texto = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
            try:
                datos = json.loads(texto)
            except json.JSONDecodeError:
                ultimo_error = f"JSON inválido (stop_reason={resp.stop_reason})"
                log.error("Respuesta no es JSON válido: %.200s", texto)
                self.uso.errores += 1
                continue
            errores = validador(datos)
            if errores:
                ultimo_error = "; ".join(errores)
                log.error("JSON con formato incorrecto: %s", ultimo_error)
                self.uso.errores += 1
                continue
            repo.cache_guardar(clave, datos, t_in, t_out)
            return datos, {
                "tokens_entrada": t_in,
                "tokens_salida": t_out,
                "cache": False,
                "hash": clave,
                "modelo": getattr(resp, "model", self.modelo),
            }
        raise ErrorAgente(f"Sin respuesta válida tras {max_reintentos + 1} intentos: {ultimo_error}")
