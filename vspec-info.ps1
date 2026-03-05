$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$scriptDir\tools\cli\vspec_info.py" @args
exit $LASTEXITCODE
