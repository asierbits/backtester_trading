@echo off
REM Arranca el panel del Laboratorio y abre el navegador.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Primero ejecuta instalar.bat
  pause
  exit /b 1
)
echo Abriendo http://localhost:8501  (cierra esta ventana para parar)
start "" http://localhost:8501
.venv\Scripts\python -m streamlit run app.py --server.port 8501
