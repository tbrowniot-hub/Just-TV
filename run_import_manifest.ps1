param([Parameter(Mandatory=$true)][string]$ManifestCsv)
$ErrorActionPreference="Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
python .\discmapper_tv_v02_1.py import-manifest --manifest $ManifestCsv --out ".\tv_index.json"
if ($LASTEXITCODE -ne 0) { throw "Python failed importing manifest (exit $LASTEXITCODE)" }
Write-Host "Wrote .\tv_index.json" -ForegroundColor Green
