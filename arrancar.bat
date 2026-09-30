@echo off
REM Arranca el panel del Laboratorio y abre el navegador cuando el servidor esta listo.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Primero ejecuta instalar.bat
  pause
  exit /b 1
)
echo Arrancando el Laboratorio en http://localhost:8501  (cierra esta ventana o pulsa Ctrl+C para parar)
echo El navegador se abrira solo en cuanto el servidor este listo...
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for ($i=0; $i -lt 60; $i++) { try { Invoke-WebRequest http://localhost:8501/_stcore/health -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process http://localhost:8501; break } catch { Start-Sleep -Seconds 1 } }"
.venv\Scripts\python -m streamlit run app.py --server.port 8501
