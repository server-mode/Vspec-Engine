$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
	& $venvPython "$scriptDir\tools\cli\vspec_run.py" --chat @args
} else {
	python "$scriptDir\tools\cli\vspec_run.py" --chat @args
}
exit $LASTEXITCODE
