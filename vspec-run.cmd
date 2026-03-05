@echo off
set SCRIPT_DIR=%~dp0
python "%SCRIPT_DIR%tools\cli\vspec_run.py" %*
exit /b %ERRORLEVEL%
