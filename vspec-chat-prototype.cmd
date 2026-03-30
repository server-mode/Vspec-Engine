@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "ACTIVATE_BAT=%SCRIPT_DIR%.venv\Scripts\activate.bat"

if not defined VSPEC_USE_VENV_PYTHON set "VSPEC_USE_VENV_PYTHON=0"
if /I "%VSPEC_USE_VENV_PYTHON%"=="1" if exist "%ACTIVATE_BAT%" (
    call "%ACTIVATE_BAT%" >nul 2>&1
)

rem Prototype defaults: ANF on for this launcher only.
if not defined VSPEC_CHAT_PROTOTYPE set "VSPEC_CHAT_PROTOTYPE=1"
if not defined VSPEC_ENABLE_ANF set "VSPEC_ENABLE_ANF=1"
if not defined VSPEC_ANF_MODE set "VSPEC_ANF_MODE=active"
if not defined VSPEC_ANF_MAX_HOT_RATIO set "VSPEC_ANF_MAX_HOT_RATIO=0.10"
if not defined VSPEC_ANF_MIN_HOT_NEURONS set "VSPEC_ANF_MIN_HOT_NEURONS=16"
if not defined VSPEC_ANF_MAX_HOT_NEURONS set "VSPEC_ANF_MAX_HOT_NEURONS=1024"
if not defined VSPEC_ANF_ACTIVATION_THRESHOLD set "VSPEC_ANF_ACTIVATION_THRESHOLD=1.10"
if not defined VSPEC_ANF_TCC_ENABLE set "VSPEC_ANF_TCC_ENABLE=1"
if not defined VSPEC_DISABLE_PY_KV_SHADOW set "VSPEC_DISABLE_PY_KV_SHADOW=1"
if not defined VSPEC_CUBLAS_CACHE_SIZE set "VSPEC_CUBLAS_CACHE_SIZE=0"
if not defined VSPEC_CHAT_MODE set "VSPEC_CHAT_MODE=python"
if not defined VSPEC_NATIVE_BACKEND set "VSPEC_NATIVE_BACKEND=python"

if not defined VSPEC_3BIT_RUNTIME_MODULE set "VSPEC_3BIT_RUNTIME_MODULE=0"
if not defined VSPEC_ULTIMATE_ENABLE set "VSPEC_ULTIMATE_ENABLE=0"
if not defined VSPEC_ULTIMATE_OUTLIER_AWARE set "VSPEC_ULTIMATE_OUTLIER_AWARE=1"
if not defined VSPEC_ULTIMATE_QLORA set "VSPEC_ULTIMATE_QLORA=0"
if not defined VSPEC_FORCE_TENSORCORE_4BIT set "VSPEC_FORCE_TENSORCORE_4BIT=0"

if /I "%~1"=="setting" (
    shift
    if not "%~1"=="" (
        :apply_loop
        if "%~1"=="" goto :print_and_exit
        call :apply_setting "%~1"
        shift
        goto :apply_loop
    )
    goto :print_and_exit
)

if "%~1"=="" (
    call "%SCRIPT_DIR%vspec-chat.cmd"
    exit /b %ERRORLEVEL%
)

call "%SCRIPT_DIR%vspec-chat.cmd" %*
exit /b %ERRORLEVEL%

:apply_setting
set "KV=%~1"
for /f "tokens=1,2 delims==:" %%A in ("%KV%") do (
    set "K=%%~A"
    set "V=%%~B"
)
if not defined V exit /b 0

if /I "%K%"=="anf" set "VSPEC_ANF_MODE=%V%"
if /I "%K%"=="anf_mode" set "VSPEC_ANF_MODE=%V%"
if /I "%K%"=="anf_hot_ratio" set "VSPEC_ANF_MAX_HOT_RATIO=%V%"
if /I "%K%"=="anf_threshold" set "VSPEC_ANF_ACTIVATION_THRESHOLD=%V%"
if /I "%K%"=="anf_tcc" set "VSPEC_ANF_TCC_ENABLE=%V%"
if /I "%K%"=="kv_shadow" set "VSPEC_DISABLE_PY_KV_SHADOW=%V%"
if /I "%K%"=="cublas_cache" set "VSPEC_CUBLAS_CACHE_SIZE=%V%"
if /I "%K%"=="chat_mode" set "VSPEC_CHAT_MODE=%V%"
if /I "%K%"=="native_backend" set "VSPEC_NATIVE_BACKEND=%V%"
if /I "%K%"=="venv_python" set "VSPEC_USE_VENV_PYTHON=%V%"
if /I "%K%"=="python" set "VSPEC_PYTHON=%V%"
if /I "%K%"=="threebit" set "VSPEC_3BIT_RUNTIME_MODULE=%V%"
if /I "%K%"=="ultimate" set "VSPEC_ULTIMATE_ENABLE=%V%"
if /I "%K%"=="outlier" set "VSPEC_ULTIMATE_OUTLIER_AWARE=%V%"
if /I "%K%"=="qlora" set "VSPEC_ULTIMATE_QLORA=%V%"
if /I "%K%"=="tensorcore" set "VSPEC_FORCE_TENSORCORE_4BIT=%V%"
exit /b 0

:print_and_exit
echo [vspec-chat-prototype] Setting list ^(option phu^)
echo.
echo   ANF ^(default ON in this launcher^)
echo     - anf / anf_mode              = off^|shadow^|active
echo     - anf_hot_ratio               = float ^(example 0.10^)
echo     - anf_threshold               = float ^(example 1.10^)
echo     - anf_tcc                     = 0^|1   ^(default 1^)
echo     - kv_shadow                   = 0^|1   ^(default 1=disable python KV shadow^)
echo     - cublas_cache                = 0..N   ^(default 0 in prototype^)
echo     - chat_mode                   = native^|python ^(default python in prototype^)
echo     - native_backend              = native-real^|python ^(default python^)
echo     - venv_python                 = 0^|1   ^(default 0; 1 = use .venv python^)
echo     - python                      = ^<path^> ^(override interpreter path^)
echo.
echo   Runtime lowbit / quality
echo     - threebit                    = 0^|1   ^(VSPEC_3BIT_RUNTIME_MODULE^)
echo     - ultimate                    = 0^|1   ^(VSPEC_ULTIMATE_ENABLE^)
echo     - outlier                     = 0^|1   ^(VSPEC_ULTIMATE_OUTLIER_AWARE^)
echo     - qlora                       = 0^|1   ^(VSPEC_ULTIMATE_QLORA^)
echo     - tensorcore                  = 0^|1   ^(VSPEC_FORCE_TENSORCORE_4BIT^)
echo.
echo Usage:
echo   vspec-chat-prototype.cmd setting
echo   vspec-chat-prototype.cmd setting anf=shadow threebit=1 ultimate=1
echo   vspec-chat-prototype.cmd --chat --max-tokens 128
echo.
echo Current effective values:
echo   VSPEC_ENABLE_ANF=%VSPEC_ENABLE_ANF%
echo   VSPEC_ANF_MODE=%VSPEC_ANF_MODE%
echo   VSPEC_ANF_MAX_HOT_RATIO=%VSPEC_ANF_MAX_HOT_RATIO%
echo   VSPEC_ANF_ACTIVATION_THRESHOLD=%VSPEC_ANF_ACTIVATION_THRESHOLD%
echo   VSPEC_ANF_TCC_ENABLE=%VSPEC_ANF_TCC_ENABLE%
echo   VSPEC_DISABLE_PY_KV_SHADOW=%VSPEC_DISABLE_PY_KV_SHADOW%
echo   VSPEC_CUBLAS_CACHE_SIZE=%VSPEC_CUBLAS_CACHE_SIZE%
echo   VSPEC_CHAT_MODE=%VSPEC_CHAT_MODE%
echo   VSPEC_NATIVE_BACKEND=%VSPEC_NATIVE_BACKEND%
echo   VSPEC_USE_VENV_PYTHON=%VSPEC_USE_VENV_PYTHON%
echo   VSPEC_PYTHON=%VSPEC_PYTHON%
echo   VSPEC_3BIT_RUNTIME_MODULE=%VSPEC_3BIT_RUNTIME_MODULE%
echo   VSPEC_ULTIMATE_ENABLE=%VSPEC_ULTIMATE_ENABLE%
echo   VSPEC_ULTIMATE_OUTLIER_AWARE=%VSPEC_ULTIMATE_OUTLIER_AWARE%
echo   VSPEC_ULTIMATE_QLORA=%VSPEC_ULTIMATE_QLORA%
echo   VSPEC_FORCE_TENSORCORE_4BIT=%VSPEC_FORCE_TENSORCORE_4BIT%
exit /b 0
