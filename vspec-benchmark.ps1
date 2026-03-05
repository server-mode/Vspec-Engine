$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$scriptDir\tools\cli\vspec_benchmark.py" @args
exit $LASTEXITCODE
