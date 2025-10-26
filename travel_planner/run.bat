@echo off
echo ========================================
echo   Travel Planner System Launcher
echo ========================================
echo.

REM Activar entorno virtual
call .venv\Scripts\activate.bat

echo [1/3] Iniciando FastAPI Server...
start "FastAPI Server" cmd /k "uvicorn app.api.server:app --reload"

timeout /t 5 /nobreak >nul

echo [2/3] Iniciando Streamlit UI...
start "Streamlit UI" cmd /k "streamlit run app/ui/streamlit_app.py"

echo.
echo ========================================
echo   Sistema iniciado correctamente!
echo ========================================
echo.
echo URLs importantes:
echo   - API Backend: http://localhost:8000
echo   - API Docs:    http://localhost:8000/docs
echo   - Frontend UI: http://localhost:8501
echo.
echo Cierra las ventanas para detener el sistema
echo ========================================

pause