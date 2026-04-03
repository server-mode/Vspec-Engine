@echo off
set SCRIPT_DIR=%~dp0
set VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe
set PY_CMD=python
if not defined VSPEC_USE_VENV_PYTHON set VSPEC_USE_VENV_PYTHON=0
if /I "%VSPEC_USE_VENV_PYTHON%"=="1" if exist "%VENV_PY%" set PY_CMD=%VENV_PY%
if defined VSPEC_PYTHON set PY_CMD=%VSPEC_PYTHON%
echo [vspec-chat] python=%PY_CMD%
if not defined VSPEC_CHAT_CLAUDE_STYLE set VSPEC_CHAT_CLAUDE_STYLE=1
if not defined VSPEC_CHAT_SHOW_PROGRESS set VSPEC_CHAT_SHOW_PROGRESS=0
if not defined VSPEC_TORCH_FORWARD set VSPEC_TORCH_FORWARD=1
if not defined VSPEC_TORCH_COMPILE set VSPEC_TORCH_COMPILE=1
if not defined VSPEC_TORCH_COMPILE_BACKEND set VSPEC_TORCH_COMPILE_BACKEND=inductor
if not defined VSPEC_TORCH_COMPILE_BACKEND_FALLBACK set VSPEC_TORCH_COMPILE_BACKEND_FALLBACK=aot_eager
if not defined VSPEC_TORCH_INDUCTOR_SMOKE set VSPEC_TORCH_INDUCTOR_SMOKE=1
if not defined VSPEC_TORCH_COMPILE_REQUIRE_TRITON set VSPEC_TORCH_COMPILE_REQUIRE_TRITON=0
if not defined VSPEC_TORCH_COMPILE_WARMUP set VSPEC_TORCH_COMPILE_WARMUP=1
if not defined VSPEC_NATIVE_CPP_LOOP_ALLOW_WITH_TORCH set VSPEC_NATIVE_CPP_LOOP_ALLOW_WITH_TORCH=1
if not defined VSPEC_CHAT_PROTOTYPE set VSPEC_CHAT_PROTOTYPE=0
if not defined VSPEC_ENABLE_ANF set VSPEC_ENABLE_ANF=0
if not defined VSPEC_CHAT_MODE set VSPEC_CHAT_MODE=python
if /I "%VSPEC_CHAT_MODE%"=="native" (
	if not defined VSPEC_NATIVE_BACKEND set VSPEC_NATIVE_BACKEND=native-real
)
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
	if /I "%VSPEC_CHAT_MODE%"=="native" if /I not "%VSPEC_NATIVE_BACKEND%"=="native-real" (
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
		"%PY_CMD%" "%SCRIPT_DIR%tools\cli\vspec_startup_menu.py"
	) else (
		"%PY_CMD%" "%SCRIPT_DIR%tools\cli\vspec_startup_menu.py"
	)
) else (
	if /I "%VSPEC_CHAT_MODE%"=="native" if /I not "%VSPEC_NATIVE_BACKEND%"=="native-real" (
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
		"%PY_CMD%" "%SCRIPT_DIR%tools\cli\vspec_run.py" --chat %*
	) else (
		"%PY_CMD%" "%SCRIPT_DIR%tools\cli\vspec_run.py" --chat %*
	)
)
exit /b %ERRORLEVEL%
