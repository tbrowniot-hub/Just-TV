$ErrorActionPreference="Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
python .\discmapper_tv_v02_1.py queue-builder --index ".\tv_index.json" --out ".\tv_queue.json"
if ($LASTEXITCODE -ne 0) { throw "Python failed launching queue builder (exit $LASTEXITCODE)" }
