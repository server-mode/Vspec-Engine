$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$scriptDir\tools\cli\vspec_server.py" @args
exit $LASTEXITCODE
