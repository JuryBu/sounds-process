param(
    [int]$BackendPort = 8017,
    [int]$FrontendPort = 5177
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Backend = Join-Path $Root "src\backend"
$Frontend = Join-Path $Root "src\frontend"
$LogDir = Join-Path $Root "outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$backendLog = Join-Path $LogDir "backend_dev.out.log"
$backendErr = Join-Path $LogDir "backend_dev.err.log"
$frontendLog = Join-Path $LogDir "frontend_dev.out.log"
$frontendErr = Join-Path $LogDir "frontend_dev.err.log"
$backendPid = Join-Path $LogDir "backend_dev.pid"
$frontendPid = Join-Path $LogDir "frontend_dev.pid"

foreach ($Port in @($BackendPort, $FrontendPort)) {
    $Listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($Listener -and $Listener.OwningProcess) {
        Stop-Process -Id $Listener.OwningProcess -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
}

Write-Host "Starting BirdVoice backend on http://127.0.0.1:$BackendPort"
$backendProcess = Start-Process -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$BackendPort" `
    -WorkingDirectory $Backend `
    -RedirectStandardOutput $backendLog `
    -RedirectStandardError $backendErr `
    -PassThru `
    -WindowStyle Hidden
$backendProcess.Id | Set-Content -Encoding UTF8 $backendPid

Write-Host "Starting BirdVoice frontend on http://127.0.0.1:$FrontendPort"
$env:VITE_API_BASE = "http://127.0.0.1:$BackendPort"
$npm = (Get-Command "npm.cmd" -ErrorAction Stop).Source
$frontendProcess = Start-Process -FilePath $npm `
    -ArgumentList "run", "dev", "--", "--host", "127.0.0.1", "--port", "$FrontendPort" `
    -WorkingDirectory $Frontend `
    -RedirectStandardOutput $frontendLog `
    -RedirectStandardError $frontendErr `
    -PassThru `
    -WindowStyle Hidden
$frontendProcess.Id | Set-Content -Encoding UTF8 $frontendPid

Start-Sleep -Seconds 4
$backendListener = Get-NetTCPConnection -LocalPort $BackendPort -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
$frontendListener = Get-NetTCPConnection -LocalPort $FrontendPort -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($backendListener -and $backendListener.OwningProcess) {
    $backendListener.OwningProcess | Set-Content -Encoding UTF8 $backendPid
}
if ($frontendListener -and $frontendListener.OwningProcess) {
    $frontendListener.OwningProcess | Set-Content -Encoding UTF8 $frontendPid
}

Write-Host "Backend PID: $(Get-Content -Encoding UTF8 $backendPid) -> $backendLog / $backendErr"
Write-Host "Frontend PID: $(Get-Content -Encoding UTF8 $frontendPid) -> $frontendLog / $frontendErr"
Write-Host "Stop with: Stop-Process -Id (Get-Content '$backendPid'),(Get-Content '$frontendPid')"
