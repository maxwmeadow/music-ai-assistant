#!/usr/bin/env pwsh

Write-Host "🚀 Starting Music AI Assistant..." -ForegroundColor Green

# Function to start a service and track its process
function Start-TrackedService {
    param($Name, $Path, $Command)
    Write-Host "Starting $Name..." -ForegroundColor Yellow

    $process = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Path'; $Command; Read-Host 'Press Enter to close'" -PassThru
    Write-Host "$Name started with PID: $($process.Id)" -ForegroundColor Gray
    return $process
}

Write-Host "Starting services in order..." -ForegroundColor Cyan

# Start Runner first
$runnerProcess = Start-TrackedService "Runner" "$PWD\runner" "npm start"
Start-Sleep 3

# Start Backend with environment variables
$backendCommand = @"
`$env:RUNNER_INGEST_URL='http://localhost:5001/eval'
`$env:ALLOWED_ORIGINS='http://localhost:3000'
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload
"@
$backendProcess = Start-TrackedService "Backend" "$PWD" $backendCommand
Start-Sleep 3

# Start Frontend in development mode
$frontendProcess = Start-TrackedService "Frontend" "$PWD\frontend" "npm run dev"

Write-Host ""
Write-Host "✅ All services are starting!" -ForegroundColor Green
Write-Host ""
Write-Host "Services:" -ForegroundColor Cyan
Write-Host "  Runner   (PID: $($runnerProcess.Id))   - http://localhost:5001" -ForegroundColor White
Write-Host "  Backend  (PID: $($backendProcess.Id))  - http://localhost:8000" -ForegroundColor White
Write-Host "  Frontend (PID: $($frontendProcess.Id)) - http://localhost:3000" -ForegroundColor White
Write-Host ""

# Create a stop script
$stopScript = @"
Write-Host "🛑 Stopping all services..." -ForegroundColor Red
try { Stop-Process -Id $($runnerProcess.Id) -Force -ErrorAction SilentlyContinue } catch {}
try { Stop-Process -Id $($backendProcess.Id) -Force -ErrorAction SilentlyContinue } catch {}
try { Stop-Process -Id $($frontendProcess.Id) -Force -ErrorAction SilentlyContinue } catch {}

# Also kill any remaining processes on these ports
Get-NetTCPConnection -LocalPort 3000,8000,5001 -ErrorAction SilentlyContinue | ForEach-Object {
    try { Stop-Process -Id `$_.OwningProcess -Force -ErrorAction SilentlyContinue } catch {}
}

Write-Host "All services stopped." -ForegroundColor Green
"@

$stopScript | Out-File -FilePath "stop-all.ps1" -Encoding UTF8

Write-Host "Created 'stop-all.ps1' to stop all services." -ForegroundColor Yellow
Write-Host ""
Write-Host "To stop all services, run: .\stop-all.ps1" -ForegroundColor Cyan
Write-Host "Or manually kill processes by PID shown above." -ForegroundColor Gray
Write-Host ""
Write-Host "Press any key to exit this script..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")