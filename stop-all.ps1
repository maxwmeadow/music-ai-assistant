Write-Host 'Stopping all services...' -ForegroundColor Red

# Kill by PID
@(10200, 14512, 23280) | ForEach-Object {
    try {
        Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
        Write-Host '  Stopped process' $_ -ForegroundColor Gray
    } catch {}
}

# Kill any remaining processes on these ports
Write-Host '  Checking ports...' -ForegroundColor Gray
Get-NetTCPConnection -LocalPort 3000,8000,5001 -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    } catch {}
}

Write-Host 'All services stopped.' -ForegroundColor Green
