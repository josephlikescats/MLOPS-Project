# Brings the whole stack up in dependency order and waits for each piece to be
# genuinely ready, because `docker compose up -d` returns long before Airflow is
# usable: the webserver and scheduler pip-install ~600MB on first boot.
# Launched by start.bat (double-click) — run directly with: powershell -File start.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$BANNER = @"
=====================================================
  E-Commerce Sales Analytics & Churn MLOps
  Starting the full stack
=====================================================
"@
Write-Host $BANNER -ForegroundColor Cyan

function Write-Step($text) { Write-Host "`n>> $text" -ForegroundColor Yellow }
function Write-Ok($text)   { Write-Host "   [OK] $text" -ForegroundColor Green }
function Write-Bad($text)  { Write-Host "   [!!] $text" -ForegroundColor Red }

function Test-Http($url) {
    try {
        $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5
        return ($r.StatusCode -eq 200)
    } catch { return $false }
}

# Polls until $Check returns true. Prints a dot per attempt so a long install
# does not look like a hang.
function Wait-Until($Label, [scriptblock]$Check, $TimeoutSec = 1200) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    Write-Host "   waiting for $Label " -NoNewline
    while ((Get-Date) -lt $deadline) {
        if (& $Check) { Write-Host ""; Write-Ok "$Label ready"; return $true }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 5
    }
    Write-Host ""
    Write-Bad "$Label did not come up within $TimeoutSec seconds"
    return $false
}

function Get-ServiceId($service, [switch]$IncludeStopped) {
    # -a is needed for airflow-init, which has already exited by the time we look.
    if ($IncludeStopped) { $id = (docker compose ps -aq $service | Select-Object -Last 1) }
    else                 { $id = (docker compose ps -q  $service | Select-Object -First 1) }
    if ($id) { return $id.Trim() }
    return ""
}

# ---------------------------------------------------------------- 1. Docker
Write-Step "Checking Docker"
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Bad "Docker daemon is not responding. Starting Docker Desktop..."
    $dd = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dd) { Start-Process $dd } else { Start-Process "Docker Desktop.exe" -ErrorAction SilentlyContinue }
    $up = Wait-Until "Docker daemon" { docker info *> $null; return ($LASTEXITCODE -eq 0) } 300
    if (-not $up) {
        Write-Bad "Could not reach Docker. Start Docker Desktop manually, then re-run."
        Read-Host "`nPress Enter to close"; exit 1
    }
}
Write-Ok "Docker is running"

# ------------------------------------------------------------- 2. Postgres
Write-Step "Starting PostgreSQL (warehouse + Airflow metadata)"
docker compose up -d postgres
if ($LASTEXITCODE -ne 0) { Write-Bad "Postgres failed to start"; Read-Host "`nPress Enter to close"; exit 1 }
$pg = Get-ServiceId postgres
$ok = Wait-Until "PostgreSQL" {
    $s = docker inspect --format "{{.State.Health.Status}}" $pg
    return ($s -eq "healthy")
} 180
if (-not $ok) { Read-Host "`nPress Enter to close"; exit 1 }

# ---------------------------------------------------------- 3. Airflow init
Write-Step "Initialising Airflow metadata database"
docker compose up -d airflow-init
$init = Get-ServiceId airflow-init -IncludeStopped
if ($init) {
    Wait-Until "Airflow init" {
        $s = docker inspect --format "{{.State.Status}}" $init
        return ($s -eq "exited")
    } 600 | Out-Null
}
Write-Ok "Airflow database ready (login: admin / admin)"

# ------------------------------------------------- 4. Airflow web+scheduler
Write-Step "Starting Airflow scheduler and webserver"
Write-Host "   First run installs ~600MB of packages inside the containers." -ForegroundColor DarkGray
Write-Host "   This can take 10-15 minutes. Later runs are much faster." -ForegroundColor DarkGray
docker compose up -d airflow-scheduler airflow-webserver
$ok = Wait-Until "Airflow webserver" { Test-Http "http://localhost:8080/health" } 1800
if (-not $ok) {
    Write-Bad "Airflow did not finish starting. Check: docker compose logs airflow-webserver"
}

# ----------------------------------------------- 5. MLflow, API, dashboard
Write-Step "Starting MLflow, the prediction API and the dashboard"
Write-Host "   (missing images are built automatically the first time)" -ForegroundColor DarkGray
docker compose up -d mlflow api dashboard
Wait-Until "MLflow"    { Test-Http "http://localhost:5000" }        600 | Out-Null
Wait-Until "API"       { Test-Http "http://localhost:8000/health" } 600 | Out-Null
Wait-Until "Dashboard" { Test-Http "http://localhost:8501" }        600 | Out-Null

# -------------------------------------------------------------- 6. The DAGs
Write-Step "Enabling the pipelines"
$sched = Get-ServiceId airflow-scheduler
if ($sched) {
    docker exec $sched airflow dags unpause retail_etl *> $null
    docker exec $sched airflow dags unpause churn_ml   *> $null
    Write-Ok "retail_etl and churn_ml are unpaused"
}

# ----------------------------------------------------------------- Summary
Write-Host @"

=====================================================
  Everything is up
=====================================================

  Dashboard     http://localhost:8501
  Airflow       http://localhost:8080     (admin / admin)
  MLflow        http://localhost:5000
  API docs      http://localhost:8000/docs
  PostgreSQL    localhost:25432           (airflow / airflow)

  Run the ETL pipeline:
    docker compose exec airflow-scheduler airflow dags trigger retail_etl

  Stop everything:
    docker compose stop

"@ -ForegroundColor Cyan

Start-Process "http://localhost:8501"
Read-Host "Press Enter to close this window"
