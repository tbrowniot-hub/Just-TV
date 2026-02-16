$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$cfgPath = Join-Path $here "config.json"
if (-not (Test-Path $cfgPath)) {
  throw "Missing config.json at $cfgPath"
}

$cfg = Get-Content -Raw $cfgPath | ConvertFrom-Json
$finalRoot = $cfg.final_tv_root
if (-not $finalRoot) {
  $finalRoot = "C:\MediaPipeline\_QUEUE\TV"
}

Write-Host "Latest TV outputs are taken from:" -ForegroundColor Cyan
Write-Host $finalRoot -ForegroundColor Green

if (Test-Path $finalRoot) {
  Write-Host ""
  Write-Host "Newest files:" -ForegroundColor Cyan
  Get-ChildItem -Path $finalRoot -Recurse -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 10 FullName, LastWriteTime
} else {
  Write-Host "Path does not exist yet." -ForegroundColor Yellow
}
