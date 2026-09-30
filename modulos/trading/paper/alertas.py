"""Alertas de paper trading por Telegram o email (opcionales, configuradas en .env).

Telegram: TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID.
Email:    SMTP_SERVIDOR, SMTP_PUERTO, SMTP_USUARIO, SMTP_CLAVE, EMAIL_DESTINO.
"""

from __future__ import annotations

import json
import smtplib
import urllib.parse
import urllib.request
from email.message import EmailMessage

from core.config import leer_env
from core.logging_setup import obtener_logger

log = obtener_logger(__name__)


def canales_configurados() -> list[str]:
    canales = []
    if leer_env("TELEGRAM_BOT_TOKEN") and leer_env("TELEGRAM_CHAT_ID"):
        canales.append("telegram")
    if leer_env("SMTP_SERVIDOR") and leer_env("EMAIL_DESTINO"):
        canales.append("email")
    return canales


def enviar_telegram(texto: str) -> bool:
    token, chat = leer_env("TELEGRAM_BOT_TOKEN"), leer_env("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    datos = urllib.parse.urlencode({"chat_id": chat, "text": texto}).encode()
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=datos, timeout=15) as r:
            return bool(json.loads(r.read()).get("ok"))
    except Exception as e:  # noqa: BLE001
        log.error("No se pudo enviar la alerta por Telegram: %s", type(e).__name__)
        return False


def enviar_email(asunto: str, texto: str) -> bool:
    servidor, destino = leer_env("SMTP_SERVIDOR"), leer_env("EMAIL_DESTINO")
    if not servidor or not destino:
        return False
    msg = EmailMessage()
    msg["Subject"] = asunto
    msg["From"] = leer_env("SMTP_USUARIO") or destino
    msg["To"] = destino
    msg.set_content(texto)
    try:
        with smtplib.SMTP(servidor, int(leer_env("SMTP_PUERTO") or 587), timeout=20) as s:
            s.starttls()
            if leer_env("SMTP_USUARIO"):
                s.login(leer_env("SMTP_USUARIO"), leer_env("SMTP_CLAVE") or "")
            s.send_message(msg)
        return True
    except Exception as e:  # noqa: BLE001
        log.error("No se pudo enviar la alerta por email: %s", type(e).__name__)
        return False


def enviar(texto: str, asunto: str = "Laboratorio: señal de paper trading") -> list[str]:
    """Envía por todos los canales configurados. Devuelve los canales que funcionaron."""
    enviados = []
    if enviar_telegram(texto):
        enviados.append("telegram")
    if enviar_email(asunto, texto):
        enviados.append("email")
    return enviados
