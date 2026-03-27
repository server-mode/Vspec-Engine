@echo off
set SCRIPT_DIR=%~dp0
set VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe
if not defined VSPEC_CHAT_MODE set VSPEC_CHAT_MODE=python
if not defined VSPEC_NATIVE_CPP_LOOP set VSPEC_NATIVE_CPP_LOOP=1
if /I "%VSPEC_CHAT_MODE%"=="native" (
	if not defined VSPEC_FULL_NATIVE_C set VSPEC_FULL_NATIVE_C=1
	if not defined VSPEC_FULL_NATIVE_BYPASS_RUNTIME set VSPEC_FULL_NATIVE_BYPASS_RUNTIME=1
	if not defined VSPEC_NATIVE_CHAT_REPL set VSPEC_NATIVE_CHAT_REPL=1
) else (
	if not defined VSPEC_FULL_NATIVE_C set VSPEC_FULL_NATIVE_C=0
	if not defined VSPEC_FULL_NATIVE_BYPASS_RUNTIME set VSPEC_FULL_NATIVE_BYPASS_RUNTIME=0
	if not defined VSPEC_NATIVE_CHAT_REPL set VSPEC_NATIVE_CHAT_REPL=0
)
if not defined VSPEC_CHAT_HARD_NATIVE set VSPEC_CHAT_HARD_NATIVE=1
if "%~1"=="" (
	if /I "%VSPEC_CHAT_MODE%"=="native" (
		if exist "%SCRIPT_DIR%build\Release\vspec_native_startup_chat.exe" (
			"%SCRIPT_DIR%build\Release\vspec_native_startup_chat.exe"
			exit /b %ERRORLEVEL%
		)
		if exist "%SCRIPT_DIR%build\Debug\vspec_native_startup_chat.exe" (
			"%SCRIPT_DIR%build\Debug\vspec_native_startup_chat.exe"
			exit /b %ERRORLEVEL%
		)
		if /I "%VSPEC_CHAT_HARD_NATIVE%"=="1" (
			echo [vspec-chat] VSPEC_CHAT_MODE=native but native startup binary not found.
			echo [vspec-chat] expected: build\Release\vspec_native_startup_chat.exe
			echo [vspec-chat] set VSPEC_CHAT_HARD_NATIVE=0 or VSPEC_CHAT_MODE=python to allow Python chat.
			exit /b 2
		)
	)
	if exist "%VENV_PY%" (
		"%VENV_PY%" "%SCRIPT_DIR%tools\cli\vspec_startup_menu.py"
	) else (
		python "%SCRIPT_DIR%tools\cli\vspec_startup_menu.py"
	)
) else (
	if /I "%VSPEC_CHAT_MODE%"=="native" (
		if exist "%SCRIPT_DIR%build\Release\vspec_native_startup_chat.exe" (
			"%SCRIPT_DIR%build\Release\vspec_native_startup_chat.exe" "%~1"
			exit /b %ERRORLEVEL%
		)
		if exist "%SCRIPT_DIR%build\Debug\vspec_native_startup_chat.exe" (
			"%SCRIPT_DIR%build\Debug\vspec_native_startup_chat.exe" "%~1"
			exit /b %ERRORLEVEL%
		)
		if /I "%VSPEC_CHAT_HARD_NATIVE%"=="1" (
			echo [vspec-chat] VSPEC_CHAT_MODE=native but native startup binary not found.
			echo [vspec-chat] set VSPEC_CHAT_HARD_NATIVE=0 or VSPEC_CHAT_MODE=python to allow Python chat.
			exit /b 2
		)
	)
	if exist "%VENV_PY%" (
		"%VENV_PY%" "%SCRIPT_DIR%tools\cli\vspec_run.py" --chat %*
	) else (
		python "%SCRIPT_DIR%tools\cli\vspec_run.py" --chat %*
	)
)
exit /b %ERRORLEVEL%
