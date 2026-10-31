# One-click demo launcher.
#
# Brings the whole stack up (if it is not already), makes sure the warehouse
# actually has data in it, then opens every UI as a browser tab in the order
# you would walk someone through them.
#
# Difference from start.ps1: start.ps1 is the "get the stack running" script.
# This one is the "I am standing in front of my teacher" script -- it is
# idempotent, skips anything already done, waits for every service to answer
# before opening a tab (so nobody sees a connection-refused page), runs the
# ETL only if the warehouse is empty, and prints a walkthrough cheat sheet.
#
# Launched by demo.bat (double-click). Direct: powershell -File demo.ps1
#
#   -NoBrowser   bring everything up and report, but do not open any tabs
#                (also skips the final "press Enter", so it can run unattended)
#
# Targets Windows PowerShell 5.1, which is what demo.bat invokes: no ternary
# "?:", no "??", no "?." anywhere in this file.

param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Banner = @"

=========================================================
  E-Commerce Sales Analytics & Churn MLOps
  Demo launcher
=========================================================
"@
Write-Host $Banner -ForegroundColor Cyan

function Write-Step($t) { Write-Host "`n>> $t" -ForegroundColor Yellow }
function Write-Ok($t)   { Write-Host "   [OK]   $t" -ForegroundColor Green }
function Write-Info($t) { Write-Host "   $t"        -ForegroundColor DarkGray }
function Write-Bad($t)  { Write-Host "   [FAIL] $t" -ForegroundColor Red }

# Every docker call goes through here.
#
# Two PowerShell 5.1 traps this exists to avoid:
#   * redirecting a native exe's stderr wraps each line in a NativeCommandError,
#     which under $ErrorActionPreference="Stop" terminates the whole script even
#     when docker exited 0;
#   * $LASTEXITCODE is the only trustworthy success signal for a native exe.
# So: stderr is swallowed with the preference relaxed locally, and the caller
# gets back an object carrying the real exit code.
function Invoke-Native {
    # NOT $Args: that is a PowerShell automatic variable, and a parameter named
    # $Args silently binds to nothing -- every docker call then runs bare.
    param([string]$Exe, [string[]]$Arguments)

    $ErrorActionPreference = "Continue"   # function-local; restores on return
    $out = & $Exe @Arguments 2>$null
    $code = $LASTEXITCODE
    if ($null -eq $out) { $text = "" } else { $text = ($out | Out-String).Trim() }
    return [pscustomobject]@{ Ok = ($code -eq 0); Code = $code; Text = $text }
}

function Invoke-Docker { param([string[]]$Arguments) return (Invoke-Native "docker" $Arguments) }

function Test-Docker {
    $r = Invoke-Docker @("info")
    return $r.Ok
}

function Test-Http($url) {
    try {
        $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5
        return ($r.StatusCode -eq 200)
    } catch { return $false }
}

# Polls until $Check returns true, printing a dot per attempt so a slow first
# boot does not look like a hang.
function Wait-Until($Label, [scriptblock]$Check, $TimeoutSec = 900) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    Write-Host "   waiting for $Label " -NoNewline
    while ((Get-Date) -lt $deadline) {
        $ok = $false
        try { $ok = (& $Check) } catch { $ok = $false }
        if ($ok) { Write-Host ""; Write-Ok "$Label ready"; return $true }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 5
    }
    Write-Host ""
    Write-Bad "$Label did not come up within $TimeoutSec seconds"
    return $false
}

function Get-ServiceId($service, [switch]$IncludeStopped) {
    # -a is needed for airflow-init, which has already exited by the time we look.
    if ($IncludeStopped) { $r = Invoke-Docker @("compose", "ps", "-aq", $service) }
    else                 { $r = Invoke-Docker @("compose", "ps", "-q",  $service) }
    if (-not $r.Ok -or -not $r.Text) { return "" }
    # @() forces an array: a single-line result would otherwise stay a string,
    # and $lines[0] would index into it and hand back one [char].
    $lines = @($r.Text -split "`r?`n" | Where-Object { $_.Trim() -ne "" })
    if ($lines.Count -eq 0) { return "" }
    if ($IncludeStopped) { return $lines[-1].Trim() }
    return $lines[0].Trim()
}

function Get-ContainerField($id, $format) {
    if (-not $id) { return "" }
    $r = Invoke-Docker @("inspect", "--format", $format, $id)
    if (-not $r.Ok) { return "" }
    return $r.Text
}

# Scalar query against the warehouse from inside the postgres container, so
# this needs neither psql nor Python on the host.
function Invoke-Scalar($sql) {
    $pg = Get-ServiceId postgres
    if (-not $pg) { return "" }
    $r = Invoke-Docker @("exec", $pg, "psql", "-U", "airflow", "-d", "retail_dw", "-tAc", $sql)
    if (-not $r.Ok) { return "" }
    return $r.Text
}

# True only for a clean non-negative integer, so a psql error message can never
# be mistaken for a row count.
function Test-Count($value) {
    return ($value -match '^\s*\d+\s*$')
}

$failed = @()

# ------------------------------------------------------------- 1. Docker
Write-Step "Checking Docker"
if (-not (Test-Docker)) {
    Write-Info "Docker is not responding -- starting Docker Desktop."
    $dd = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dd) { Start-Process $dd }
    else { Start-Process "Docker Desktop.exe" -ErrorAction SilentlyContinue }
    if (-not (Wait-Until "Docker daemon" { Test-Docker } 300)) {
        Write-Bad "Could not reach Docker. Start Docker Desktop manually, then re-run."
        if (-not $NoBrowser) { Read-Host "`nPress Enter to close" }
        exit 1
    }
} else {
    Write-Ok "Docker is running"
}

# ----------------------------------------------------------- 2. Postgres
Write-Step "PostgreSQL (warehouse + Airflow metadata)"
$r = Invoke-Docker @("compose", "up", "-d", "postgres")
if (-not $r.Ok) {
    Write-Bad "Could not start Postgres:`n$($r.Text)"
    if (-not $NoBrowser) { Read-Host "`nPress Enter to close" }
    exit 1
}
$pg = Get-ServiceId postgres
if (-not (Wait-Until "PostgreSQL" {
        return ((Get-ContainerField $pg "{{.State.Health.Status}}") -eq "healthy")
    } 240)) {
    Write-Bad "Postgres never became healthy. Check: docker compose logs postgres"
    if (-not $NoBrowser) { Read-Host "`nPress Enter to close" }
    exit 1
}

# ------------------------------------------------------- 3. Airflow init
# Idempotent: `airflow db migrate` on an initialised database is a no-op, and
# the admin-user create is guarded with `|| true` in compose.
Write-Step "Airflow metadata database"
$initId = Get-ServiceId airflow-init -IncludeStopped
$needInit = $true
if ($initId) {
    $st = Get-ContainerField $initId "{{.State.Status}}:{{.State.ExitCode}}"
    if ($st -eq "exited:0") { $needInit = $false; Write-Ok "already initialised" }
}
if ($needInit) {
    Invoke-Docker @("compose", "up", "-d", "airflow-init") | Out-Null
    $initId = Get-ServiceId airflow-init -IncludeStopped
    Wait-Until "Airflow init" {
        return ((Get-ContainerField $initId "{{.State.Status}}") -eq "exited")
    } 1200 | Out-Null
    Write-Ok "Airflow database ready (login: admin / admin)"
}

# -------------------------------------------- 4. Airflow web + scheduler
Write-Step "Airflow scheduler and webserver"
Write-Info "First boot pip-installs ~600MB inside the containers (10-15 min)."
Write-Info "Afterwards this step takes well under a minute."
Invoke-Docker @("compose", "up", "-d", "airflow-scheduler", "airflow-webserver") | Out-Null
if (-not (Wait-Until "Airflow webserver" { Test-Http "http://localhost:8080/health" } 2400)) {
    Write-Bad "Airflow did not finish starting. Check: docker compose logs airflow-webserver"
    $failed += "Airflow"
}

# ------------------------------------------ 5. MLflow, API and dashboard
Write-Step "MLflow, prediction API and dashboard"
Write-Info "Missing images are built automatically the first time."
Invoke-Docker @("compose", "up", "-d", "mlflow", "api", "dashboard") | Out-Null
if (-not (Wait-Until "MLflow"    { Test-Http "http://localhost:5000" }                 900)) { $failed += "MLflow" }
if (-not (Wait-Until "API"       { Test-Http "http://localhost:8000/health" }           900)) { $failed += "API" }
if (-not (Wait-Until "Dashboard" { Test-Http "http://localhost:8501/_stcore/health" }   900)) { $failed += "Dashboard" }

# --------------------------------------------------------- 6. The DAGs on
Write-Step "Enabling the pipelines"
$sched = Get-ServiceId airflow-scheduler
if ($sched) {
    Invoke-Docker @("exec", $sched, "airflow", "dags", "unpause", "retail_etl") | Out-Null
    Invoke-Docker @("exec", $sched, "airflow", "dags", "unpause", "churn_ml")   | Out-Null
    Write-Ok "retail_etl and churn_ml are unpaused"
} else {
    Write-Bad "No scheduler container -- cannot unpause the DAGs"
    $failed += "scheduler"
}

# ------------------------------------------------------------ 7. The data
# An empty warehouse means empty dashboard charts, which is the one thing you
# do not want to discover in front of an audience. The ETL runs only if the
# fact table is actually empty, so a repeat demo starts in seconds.
Write-Step "Checking the warehouse has data"
$factRows = Invoke-Scalar "select count(*) from dw.fact_sales;"
if ((Test-Count $factRows) -and [int64]$factRows -gt 0) {
    Write-Ok ("dw.fact_sales already holds {0:N0} rows -- skipping the ETL" -f [int64]$factRows)
} elseif (-not $sched) {
    Write-Bad "Warehouse is empty and there is no scheduler to run the ETL"
    $failed += "retail_etl"
} else {
    Write-Info "Warehouse is empty. Triggering retail_etl (~5-10 min on 1.07M rows)."
    Invoke-Docker @("exec", $sched, "airflow", "dags", "trigger", "retail_etl") | Out-Null
    Wait-Until "retail_etl to load the warehouse" {
        $n = Invoke-Scalar "select count(*) from dw.fact_sales;"
        return ((Test-Count $n) -and [int64]$n -gt 0)
    } 3000 | Out-Null

    $factRows = Invoke-Scalar "select count(*) from dw.fact_sales;"
    if ((Test-Count $factRows) -and [int64]$factRows -gt 0) {
        Write-Ok ("warehouse loaded: {0:N0} fact rows" -f [int64]$factRows)
    } else {
        Write-Bad "ETL did not populate the warehouse. Check the run at http://localhost:8080"
        $failed += "retail_etl"
    }
}

# ------------------------------------------------------- 8. What is loaded
Write-Step "Warehouse contents"
$counts = @(
    @{ Label = "raw.sales_raw";             Sql = "select count(*) from raw.sales_raw;" },
    @{ Label = "staging.sales_valid";       Sql = "select count(*) from staging.sales_valid;" },
    @{ Label = "staging.rejected_records";  Sql = "select count(*) from staging.rejected_records;" },
    @{ Label = "dw.fact_sales";             Sql = "select count(*) from dw.fact_sales;" },
    @{ Label = "mart.customer_rfm";         Sql = "select count(*) from mart.customer_rfm;" },
    @{ Label = "mart.sales_monthly";        Sql = "select count(*) from mart.sales_monthly;" }
)
foreach ($c in $counts) {
    $v = Invoke-Scalar $c.Sql
    if (Test-Count $v) { Write-Info ("{0,-26} {1,12:N0}" -f $c.Label, [int64]$v) }
    else               { Write-Info ("{0,-26} {1,12}"    -f $c.Label, "unavailable") }
}

# ----------------------------------------------------------- 9. Open tabs
Write-Step "Opening the demo tabs"
# Walkthrough order; the dashboard is last so it lands on top as the starting point.
$tabs = @(
    @{ Url = "http://localhost:5000";      Name = "MLflow    - experiments & model registry" },
    @{ Url = "http://localhost:8000/docs"; Name = "API       - interactive Swagger docs" },
    @{ Url = "http://localhost:8080/dags"; Name = "Airflow   - the two DAGs" },
    @{ Url = "http://localhost:8501";      Name = "Dashboard - start here" }
)
foreach ($t in $tabs) {
    if ($NoBrowser) {
        Write-Info ("would open  {0}" -f $t.Name)
    } else {
        Write-Info ("opening     {0}" -f $t.Name)
        Start-Process $t.Url
        Start-Sleep -Milliseconds 1200   # keeps tab order predictable
    }
}

# ------------------------------------------------------------ 10. Summary
if ($failed.Count -gt 0) {
    Write-Host "`n  WARNING: these did not come up cleanly: $($failed -join ', ')" -ForegroundColor Red
    Write-Host "  Diagnose with:  docker compose ps   /   docker compose logs <service>" -ForegroundColor Red
}

$Sheet = @"

=========================================================
  Everything is up -- walkthrough cheat sheet
=========================================================

  1. Dashboard      http://localhost:8501
       Sales/order trend, top products, customer
       segmentation, country sales, retention, churn
       scores, and the data-quality panel.

  2. Airflow        http://localhost:8080        admin / admin
       retail_etl : extract >> validate >> transform
       churn_ml   : monitor >> decide >> retrain >> score >> verify
       Worth saying out loud: churn_ml short-circuits the
       retrain unless a monitoring criterion actually fires.

  3. MLflow         http://localhost:5000
       All three candidate models logged per run, and
       'retail-churn-classifier' in the model registry.

  4. API docs       http://localhost:8000/docs
       POST /predict via the "Try it out" button.
       GET /metrics shows request counts and p50/p95/p99.

  PostgreSQL        localhost:25432              airflow / airflow

  Useful one-liners
  -----------------
  Re-run the ETL          docker compose exec airflow-scheduler airflow dags trigger retail_etl
  Run the churn pipeline  docker compose exec airflow-scheduler airflow dags trigger churn_ml
  Service status          docker compose ps
  Stop everything         docker compose stop
  Reset completely        docker compose down -v

"@
Write-Host $Sheet -ForegroundColor Cyan

# No ternary below: Windows PowerShell 5.1 has no "?:" operator.
if ($NoBrowser) {
    if ($failed.Count -gt 0) { exit 1 } else { exit 0 }
}

# Double-clicked through demo.bat the console closes the instant this exits,
# and this window is the one showing the cheat sheet -- so hold it open.
Read-Host "Press Enter to close this window (the stack keeps running)"
if ($failed.Count -gt 0) { exit 1 }
