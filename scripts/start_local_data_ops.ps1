# Canonical local Data Ops launcher — http://127.0.0.1:8510 only.
# Password: export PBJ_DATA_OPS_PASSWORD, or _scratch/data_ops.local.env (never commit).

$ErrorActionPreference = "Stop"
$CanonicalPort = 8510
$CanonicalUrl = "http://127.0.0.1:$CanonicalPort/"

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

if ($env:PBJ_DATA_OPS_PORT -and [int]$env:PBJ_DATA_OPS_PORT -ne $CanonicalPort) {
    Write-Error "Data Ops local URL is fixed at $CanonicalUrl. Unset PBJ_DATA_OPS_PORT (was $($env:PBJ_DATA_OPS_PORT))."
    exit 1
}

$env:PBJ_DATA_OPS_PORT = "$CanonicalPort"

if (-not ($env:PRIVATE_DATA_ROOT -and $env:PRIVATE_DATA_ROOT.Trim())) {
    $PrivateFallback = "D:\Data\PBJ320_Private"
    if (Test-Path -LiteralPath $PrivateFallback) {
        $env:PRIVATE_DATA_ROOT = $PrivateFallback
    }
}

if (-not $env:PBJ_ACTIVE_RELEASE_REGISTRY) {
    $env:PBJ_ACTIVE_RELEASE_REGISTRY = Join-Path $RepoRoot "state\active_releases.json"
}
if (-not $env:PBJ_REPO_ROOT) {
    $PbjApp = Join-Path $RepoRoot "..\PBJapp"
    if (-not (Test-Path $PbjApp)) {
        Write-Error "PBJ_REPO_ROOT is not set and default path not found: $PbjApp"
        exit 1
    }
    $env:PBJ_REPO_ROOT = (Resolve-Path $PbjApp).Path
}

# Facility deploy bundles live under PBJ_DATA_ROOT (not PBJapp\deployments stubs).
if (-not ($env:PBJ_DATA_ROOT -and $env:PBJ_DATA_ROOT.Trim())) {
    $DataRootFallback = "D:\PBJapp-data"
    if (Test-Path -LiteralPath $DataRootFallback) {
        $env:PBJ_DATA_ROOT = $DataRootFallback
    }
}

if (-not ($env:PBJ_DATA_OPS_PASSWORD -and $env:PBJ_DATA_OPS_PASSWORD.Trim())) {
    Write-Error @"
PBJ_DATA_OPS_PASSWORD is not set.
  `$env:PBJ_DATA_OPS_PASSWORD = '<your-password>'
  or add PBJ_DATA_OPS_PASSWORD=... to _scratch/data_ops.local.env
"@
    exit 1
}

$listeners = @(Get-NetTCPConnection -LocalPort $CanonicalPort -State Listen -ErrorAction SilentlyContinue)
if ($listeners.Count -gt 0) {
    Write-Host "ERROR: port $CanonicalPort is already in use. Data Ops was not started." -ForegroundColor Red
    $seen = @{}
    foreach ($conn in $listeners) {
        $ownerPid = [int]$conn.OwningProcess
        if ($seen.ContainsKey($ownerPid)) { continue }
        $seen[$ownerPid] = $true
        $proc = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
        $name = if ($proc) { $proc.ProcessName } else { "unknown" }
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$ownerPid" -ErrorAction SilentlyContinue).CommandLine
        Write-Host "  PID $ownerPid ($name)"
        if ($cmd) { Write-Host "    $cmd" }
        Write-Host "  Stop: Stop-Process -Id $ownerPid -Force"
    }
    Write-Host "Then rerun: .\scripts\start_local_data_ops.ps1"
    exit 1
}

Write-Host "Data Ops URL: $CanonicalUrl"
Write-Host "Registry:     $env:PBJ_ACTIVE_RELEASE_REGISTRY"
Write-Host "PBJapp root:  $env:PBJ_REPO_ROOT"
Write-Host "Data root:    $env:PBJ_DATA_ROOT"
python data_ops_app.py
