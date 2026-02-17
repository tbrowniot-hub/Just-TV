$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
python -m uvicorn tv_ui_app:app --host 127.0.0.1 --port 8787
