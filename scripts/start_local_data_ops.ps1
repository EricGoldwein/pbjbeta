# Local Data Ops launcher — kills stale listener, then starts Flask on 8510.
# Password: set PBJ_DATA_OPS_PASSWORD in the shell, or create _scratch/data_ops.local.env (never commit).

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

$LocalEnv = Join-Path $RepoRoot "_scratch\data_ops.local.env"
if (Test-Path $LocalEnv) {
    Get-Content $LocalEnv | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line -match "^([^=]+)=(.*)$") {
            $name = $Matches[1].Trim()
            $value = $Matches[2].Trim().Trim('"')
            Set-Item -Path "Env:$name" -Value $value
        }
    }
}

if (-not $env:PBJ_ACTIVE_RELEASE_REGISTRY) {
    $env:PBJ_ACTIVE_RELEASE_REGISTRY = Join-Path $RepoRoot "state\active_releases.json"
}
if (-not $env:PBJ_REPO_ROOT) {
    $env:PBJ_REPO_ROOT = (Resolve-Path (Join-Path $RepoRoot "..\PBJapp")).Path
}
if (-not $env:PBJ_DATA_OPS_PORT) {
    $env:PBJ_DATA_OPS_PORT = "8510"
}

$port = [int]$env:PBJ_DATA_OPS_PORT
Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }

Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction SilentlyContinue).CommandLine
    if ($cmd -like "*data_ops_app*") {
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
}

if (-not ($env:PBJ_DATA_OPS_PASSWORD -and $env:PBJ_DATA_OPS_PASSWORD.Trim())) {
    Write-Error "PBJ_DATA_OPS_PASSWORD is not set. Export it or add to _scratch/data_ops.local.env"
    exit 1
}

Write-Host "Starting Data Ops on http://127.0.0.1:$port/ (registry: $env:PBJ_ACTIVE_RELEASE_REGISTRY)"
python data_ops_app.py
