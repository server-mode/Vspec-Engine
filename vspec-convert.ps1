$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$scriptDir\tools\cli\vspec_convert.py" @args
exit $LASTEXITCODE
