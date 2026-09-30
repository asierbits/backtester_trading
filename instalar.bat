@echo off
REM Instala el Laboratorio en Windows: crea el entorno virtual e instala las dependencias.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  set PY=py -3
) else (
  set PY=python
)
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
  echo Hace falta Python 3.11 o superior. Descargalo de https://www.python.org/downloads/
  echo Marca la casilla "Add Python to PATH" al instalarlo.
  pause
  exit /b 1
)
%PY% -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if not exist .env (
  copy .env.example .env >nul
  echo Creado .env: pon tu ANTHROPIC_API_KEY ahi ^(o en la pagina Ajustes^) si quieres usar los agentes.
)
echo.
echo Instalacion terminada. Para arrancar: arrancar.bat
pause
