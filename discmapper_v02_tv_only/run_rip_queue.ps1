$ErrorActionPreference="Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
python .\discmapper_tv_v02.py rip-queue --index ".\tv_index.json" --queue ".\tv_queue.json" --config ".\config.json"
if ($LASTEXITCODE -ne 0) { throw "Python rip-queue exited $LASTEXITCODE" }