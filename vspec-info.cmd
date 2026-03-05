@echo off
set SCRIPT_DIR=%~dp0
python "%SCRIPT_DIR%tools\cli\vspec_info.py" %*
exit /b %ERRORLEVEL%
