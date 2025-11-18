@echo off
chcp 65001 >nul
cls

echo ====================================================
echo   Travel Planner - Instalacion y Ejecucion
echo ====================================================
echo.

REM Verificar que estamos en el directorio correcto
if not exist "app" (
    echo [ERROR] No se encuentra la carpeta app
    echo Ejecuta este script desde la carpeta travel_planner
    pause
    exit /b 1
)

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta instalado
    pause
    exit /b 1
)

echo [OK] Python encontrado
python --version
echo.

REM Verificar entorno virtual
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creando entorno virtual...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual
        pause
        exit /b 1
    )
    echo [OK] Entorno virtual creado
) else (
    echo [OK] Entorno virtual encontrado
)
echo.

REM Activar entorno virtual
echo [INFO] Activando entorno virtual...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] No se pudo activar el entorno virtual
    pause
    exit /b 1
)
echo [OK] Entorno virtual activado
echo.

REM Verificar requirements.txt
if not exist "requirements.txt" (
    echo [ERROR] No se encuentra requirements.txt
    pause
    exit /b 1
)

REM Actualizar pip
echo [INFO] Actualizando pip...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet

REM Instalar dependencias
echo.
echo [INFO] Instalando dependencias desde requirements.txt...
echo [INFO] Esto puede tardar varios minutos en la primera ejecucion...
echo.
.venv\Scripts\python.exe -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Fallo la instalacion de dependencias
    echo Verifica que requirements.txt este correcto
    pause
    exit /b 1
)

echo.
echo [OK] Dependencias instaladas correctamente
echo.

REM Verificar instalacion de modulos criticos
echo [INFO] Verificando modulos criticos...
.venv\Scripts\python.exe -c "import uvicorn; print('  [OK] uvicorn')" 2>nul
if errorlevel 1 echo   [ERROR] uvicorn no instalado

.venv\Scripts\python.exe -c "import streamlit; print('  [OK] streamlit')" 2>nul
if errorlevel 1 echo   [ERROR] streamlit no instalado

.venv\Scripts\python.exe -c "import fastapi; print('  [OK] fastapi')" 2>nul
if errorlevel 1 echo   [ERROR] fastapi no instalado

.venv\Scripts\python.exe -c "import sklearn; print('  [OK] scikit-learn')" 2>nul
if errorlevel 1 echo   [ERROR] scikit-learn no instalado

echo.
echo ====================================================
echo   Iniciando Servicios
echo ====================================================
echo.

REM Iniciar FastAPI Server (Sin --reload para persistencia)
echo [1/2] Iniciando FastAPI Server en puerto 8000...
start "FastAPI Server - Travel Planner" cmd /k "title FastAPI Server && cd /d "%CD%" && call .venv\Scripts\activate.bat && echo Iniciando API... && .venv\Scripts\python.exe -m uvicorn app.api.server:app --host 0.0.0.0 --port 8000"

echo [INFO] Esperando 12 segundos para que la API inicie...
timeout /t 12 /nobreak >nul

REM Iniciar Streamlit UI
echo [2/2] Iniciando Streamlit UI en puerto 8501...
start "Streamlit UI - Travel Planner" cmd /k "title Streamlit UI && cd /d "%CD%" && call .venv\Scripts\activate.bat && echo Iniciando UI... && .venv\Scripts\python.exe -m streamlit run app\ui\streamlit_app.py"

echo.
echo ====================================================
echo   Sistema Iniciado Correctamente!
echo ====================================================
echo.
echo  URLs Importantes:
echo    API Backend:   http://localhost:8000
echo    API Docs:      http://localhost:8000/docs
echo    Frontend UI:   http://localhost:8501
echo.
echo  Se abrieron 2 ventanas nuevas:
echo    1. FastAPI Server (Backend)
echo    2. Streamlit UI (Frontend)
echo.
echo  IMPORTANTE: NO cierres esas ventanas mientras uses el sistema
echo.
echo  Para detener: Cierra las ventanas o presiona Ctrl+C en cada una
echo.
echo ====================================================

REM (Eliminado el bloque que abría el navegador manualmente)
echo Presiona cualquier tecla para cerrar esta ventana
echo (Los servicios seguiran ejecutandose)
pause >nul

exit /b 0