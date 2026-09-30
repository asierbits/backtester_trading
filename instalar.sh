#!/usr/bin/env bash
# Instala el Laboratorio en Mac o Linux: crea el entorno virtual e instala las dependencias.
set -e
cd "$(dirname "$0")"

PY=""
for candidato in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidato" >/dev/null 2>&1; then
    if "$candidato" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
      PY="$candidato"; break
    fi
  fi
done
if [ -z "$PY" ]; then
  echo "Hace falta Python 3.11 o superior. Descárgalo de https://www.python.org/downloads/"
  exit 1
fi

echo "Usando $($PY --version)"
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Creado .env: pon tu ANTHROPIC_API_KEY ahí (o en la página Ajustes) si quieres usar los agentes."
fi
echo ""
echo "Instalación terminada. Para arrancar:  ./arrancar.sh"
