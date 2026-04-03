$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (-not $env:VSPEC_CHAT_CLAUDE_STYLE) { $env:VSPEC_CHAT_CLAUDE_STYLE = "1" }
if (-not $env:VSPEC_CHAT_SHOW_PROGRESS) { $env:VSPEC_CHAT_SHOW_PROGRESS = "0" }
if (-not $env:VSPEC_TORCH_FORWARD) { $env:VSPEC_TORCH_FORWARD = "1" }
if (-not $env:VSPEC_TORCH_COMPILE) { $env:VSPEC_TORCH_COMPILE = "1" }
if (-not $env:VSPEC_TORCH_COMPILE_BACKEND) { $env:VSPEC_TORCH_COMPILE_BACKEND = "inductor" }
if (-not $env:VSPEC_TORCH_COMPILE_BACKEND_FALLBACK) { $env:VSPEC_TORCH_COMPILE_BACKEND_FALLBACK = "aot_eager" }
if (-not $env:VSPEC_TORCH_INDUCTOR_SMOKE) { $env:VSPEC_TORCH_INDUCTOR_SMOKE = "1" }
if (-not $env:VSPEC_TORCH_COMPILE_REQUIRE_TRITON) { $env:VSPEC_TORCH_COMPILE_REQUIRE_TRITON = "0" }
if (-not $env:VSPEC_TORCH_COMPILE_WARMUP) { $env:VSPEC_TORCH_COMPILE_WARMUP = "1" }
if (-not $env:VSPEC_NATIVE_CPP_LOOP_ALLOW_WITH_TORCH) { $env:VSPEC_NATIVE_CPP_LOOP_ALLOW_WITH_TORCH = "1" }
if (Test-Path $venvPython) {
	& $venvPython "$scriptDir\tools\cli\vspec_run.py" --chat @args
} else {
	python "$scriptDir\tools\cli\vspec_run.py" --chat @args
}
exit $LASTEXITCODE
