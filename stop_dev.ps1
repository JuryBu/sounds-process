$ErrorActionPreference = "SilentlyContinue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $Root "outputs\logs"
$PidFiles = @(
    Join-Path $LogDir "backend_dev.pid"
    Join-Path $LogDir "frontend_dev.pid"
)

foreach ($PidFile in $PidFiles) {
    if (Test-Path $PidFile) {
        $ProcessId = Get-Content -Encoding UTF8 $PidFile
        if ($ProcessId) {
            Stop-Process -Id $ProcessId -Force
            Write-Host "Stopped PID $ProcessId"
        }
        Remove-Item -LiteralPath $PidFile -Force
    }
}

foreach ($Port in @(8017, 5177)) {
    $Listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($Listener -and $Listener.OwningProcess) {
        Stop-Process -Id $Listener.OwningProcess -Force
        Write-Host "Stopped listener on port $Port PID $($Listener.OwningProcess)"
    }
}
