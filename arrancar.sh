#!/usr/bin/env bash
# Arranca el panel del Laboratorio y abre el navegador.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Primero ejecuta ./instalar.sh"
  exit 1
fi
PUERTO="${PUERTO:-8501}"
URL="http://localhost:$PUERTO"
echo "Abriendo $URL  (Ctrl+C para parar)"
( sleep 3; if command -v open >/dev/null; then open "$URL"; elif command -v xdg-open >/dev/null; then xdg-open "$URL"; fi ) >/dev/null 2>&1 &
exec .venv/bin/python -m streamlit run app.py --server.port "$PUERTO"
