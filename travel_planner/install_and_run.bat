@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "INSTALLED_MARKER=.venv\requirements.installed"

echo ====================================================
echo   Travel Planner
echo ====================================================

if not exist "%PY%" (
    echo [INFO] Creando entorno virtual...
    python -m venv .venv || goto :fail
)

fc /b requirements.txt "%INSTALLED_MARKER%" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Instalando dependencias, puede tardar unos minutos...
    "%PY%" -m pip install --quiet --upgrade pip || goto :fail
    "%PY%" -m pip install --quiet -r requirements.txt || goto :fail
    copy /y requirements.txt "%INSTALLED_MARKER%" >nul
) else (
    echo [OK] Dependencias al dia
)

"%PY%" verify_setup.py || goto :fail

call :stop_api
call :port_in_use 8000 && (echo [FALLO] El puerto 8000 ya esta en uso. Cerra la otra instancia. & goto :fail)
call :port_in_use 8501 && (echo [FALLO] El puerto 8501 ya esta en uso. Cerra la otra instancia. & goto :fail)

echo [INFO] Iniciando API...
start "" /b "%PY%" -m uvicorn app.api.server:app --host 127.0.0.1 --port 8000 --log-level warning

"%PY%" verify_setup.py --api || goto :fail_with_api

echo.
echo ====================================================
echo   Todo listo
echo     Interfaz:  http://localhost:8501
echo     API docs:  http://localhost:8000/docs
echo   Ctrl+C para detener todo
echo ====================================================
echo.

start "" /b cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:8501"
"%PY%" -m streamlit run app\ui\streamlit_app.py --server.headless true --browser.gatherUsageStats false

call :stop_api
exit /b 0

:port_in_use
netstat -ano | findstr /r /c:":%1 .*LISTENING" >nul
exit /b %errorlevel%

:stop_api
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*uvicorn app.api.server:app*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
exit /b 0

:fail_with_api
call :stop_api

:fail
echo.
echo [FALLO] No se pudo iniciar Travel Planner. Revisa los mensajes de arriba.
pause
exit /b 1
