Write-Host "ðŸ›‘ Stopping all services..." -ForegroundColor Red
try { Stop-Process -Id 38508 -Force -ErrorAction SilentlyContinue } catch {}
try { Stop-Process -Id 16996 -Force -ErrorAction SilentlyContinue } catch {}
try { Stop-Process -Id 33356 -Force -ErrorAction SilentlyContinue } catch {}

# Also kill any remaining processes on these ports
Get-NetTCPConnection -LocalPort 3000,8000,5001 -ErrorAction SilentlyContinue | ForEach-Object {
    try { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } catch {}
}

Write-Host "All services stopped." -ForegroundColor Green
