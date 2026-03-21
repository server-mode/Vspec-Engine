@echo off
set SCRIPT_DIR=%~dp0
set VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe
if "%~1"=="" (
	if exist "%VENV_PY%" (
		"%VENV_PY%" "%SCRIPT_DIR%tools\cli\vspec_startup_menu.py"
	) else (
		python "%SCRIPT_DIR%tools\cli\vspec_startup_menu.py"
	)
) else (
	if exist "%VENV_PY%" (
		"%VENV_PY%" "%SCRIPT_DIR%tools\cli\vspec_info.py" %*
	) else (
		python "%SCRIPT_DIR%tools\cli\vspec_info.py" %*
	)
)
exit /b %ERRORLEVEL%
