Write-Host "ðŸ›‘ Stopping all services..." -ForegroundColor Red
try { Stop-Process -Id 22824 -Force -ErrorAction SilentlyContinue } catch {}
try { Stop-Process -Id 22924 -Force -ErrorAction SilentlyContinue } catch {}
try { Stop-Process -Id 22852 -Force -ErrorAction SilentlyContinue } catch {}

# Also kill any remaining processes on these ports
Get-NetTCPConnection -LocalPort 3000,8000,5001 -ErrorAction SilentlyContinue | ForEach-Object {
    try { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } catch {}
}

Write-Host "All services stopped." -ForegroundColor Green
