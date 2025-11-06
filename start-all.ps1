#!/usr/bin/env pwsh

Write-Host "Starting Music AI Assistant..." -ForegroundColor Green
Write-Host ""

# Function to start a service in a new window
function Start-Service {
    param($Name, $Path, $ScriptBlock)
    Write-Host "Starting $Name..." -ForegroundColor Yellow

    $script = [scriptblock]::Create($ScriptBlock)
    $process = Start-Process powershell -ArgumentList "-NoExit", "-Command", $ScriptBlock -PassThru

    Write-Host "  OK $Name started (PID: $($process.Id))" -ForegroundColor Green
    return $process
}

Write-Host "Starting services in order..." -ForegroundColor Cyan
Write-Host ""

# Start Runner first
Write-Host "[1/3] Runner" -ForegroundColor Cyan
$runnerCmd = "cd '$PWD\runner'; Write-Host 'Starting Runner...' -ForegroundColor Yellow; npm start; Read-Host 'Press Enter to close'"
$runnerProcess = Start-Service "Runner" "$PWD\runner" $runnerCmd
Start-Sleep -Seconds 3

# Start Backend with proper venv activation
Write-Host "[2/3] Backend" -ForegroundColor Cyan
$backendCmd = @"
Write-Host 'Starting Backend...' -ForegroundColor Yellow
cd '$PWD'
Set-Location backend
Write-Host 'Activating virtual environment...' -ForegroundColor Gray
& '.\.venv\Scripts\Activate.ps1'
Write-Host 'Setting environment variables...' -ForegroundColor Gray
`$env:RUNNER_INGEST_URL = 'http://localhost:5001/eval'
`$env:ALLOWED_ORIGINS = 'http://localhost:3000'
Write-Host 'Verifying PyTorch is installed...' -ForegroundColor Gray
python -c 'import torch; print(torch.__version__)'
Set-Location ..
Write-Host 'Starting uvicorn...' -ForegroundColor Yellow
uvicorn backend.main:app --reload
Read-Host 'Press Enter to close'
"@
$backendProcess = Start-Service "Backend" "$PWD" $backendCmd
Start-Sleep -Seconds 3

# Start Frontend
Write-Host "[3/3] Frontend" -ForegroundColor Cyan
$frontendCmd = "cd '$PWD\frontend'; Write-Host 'Starting Frontend...' -ForegroundColor Yellow; npm run dev; Read-Host 'Press Enter to close'"
$frontendProcess = Start-Service "Frontend" "$PWD\frontend" $frontendCmd

Write-Host ""
Write-Host "All services started successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Services Running:" -ForegroundColor Cyan
Write-Host "  Runner   (PID: $($runnerProcess.Id))  -> http://localhost:5001" -ForegroundColor White
Write-Host "  Backend  (PID: $($backendProcess.Id))  -> http://localhost:8000" -ForegroundColor White
Write-Host "  Frontend (PID: $($frontendProcess.Id)) -> http://localhost:3000" -ForegroundColor White
Write-Host ""

# Create stop script
$stopScript = @"
Write-Host 'Stopping all services...' -ForegroundColor Red

# Kill by PID
@($($runnerProcess.Id), $($backendProcess.Id), $($frontendProcess.Id)) | ForEach-Object {
    try {
        Stop-Process -Id `$_ -Force -ErrorAction SilentlyContinue
        Write-Host '  Stopped process' `$_ -ForegroundColor Gray
    } catch {}
}

# Kill any remaining processes on these ports
Write-Host '  Checking ports...' -ForegroundColor Gray
Get-NetTCPConnection -LocalPort 3000,8000,5001 -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        Stop-Process -Id `$_.OwningProcess -Force -ErrorAction SilentlyContinue
    } catch {}
}

Write-Host 'All services stopped.' -ForegroundColor Green
"@

$stopScript | Out-File -FilePath "stop-all.ps1" -Encoding UTF8

Write-Host "Created 'stop-all.ps1' to stop all services" -ForegroundColor Yellow
Write-Host ""
Write-Host "To stop all services, run:" -ForegroundColor Cyan
Write-Host "  .\stop-all.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to exit this launcher..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")