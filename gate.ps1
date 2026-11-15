param(
  [switch]$Full,
  [string]$LogRoot = "logs/gate"
)

$ErrorActionPreference = "Continue"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$stamp = Get-Date -Format "yyyy.MM.dd-HHmmss"
$gateDir = Join-Path $LogRoot $stamp
New-Item -ItemType Directory -Force -Path $gateDir | Out-Null
$summaryFile = Join-Path $gateDir "summary.log"

"gate start {0} Full={1}" -f $stamp, $Full | Tee-Object -FilePath $summaryFile

function Invoke-GateStep {
  param(
    [string]$StepName,
    [scriptblock]$Action
  )
  $logFile = Join-Path $gateDir ($StepName + ".log")
  New-Item -ItemType File -Force -Path $logFile | Out-Null
  Write-Host ("[gate] step {0}" -f $StepName)
  $started = Get-Date
  & $Action 2>&1 | Tee-Object -FilePath $logFile
  $code = $LASTEXITCODE
  $secs = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
  $line = "{0} exit={1} elapsed={2}s" -f $StepName, $code, $secs
  $line | Add-Content -Path $summaryFile
  Write-Host ("[gate] {0}" -f $line)
  if ($code -ne 0) {
    Write-Host ("[gate] FAIL at {0} code {1}" -f $StepName, $code)
    exit $code
  }
}

Invoke-GateStep "fmt" { cargo fmt --check }
Invoke-GateStep "clippy" { cargo clippy --workspace --all-targets -- -D warnings }
Invoke-GateStep "test" { cargo test --workspace }
Invoke-GateStep "release-build" { cargo build --release }
Invoke-GateStep "bench-smoke" { .\target\release\graymul-bench.exe --tier tiny --task-id gate --output (Join-Path $gateDir "bench-tiny.json") }

if ($Full) {
  $maturinProbe = Get-Command maturin -ErrorAction SilentlyContinue
  if ($maturinProbe -eq $null) {
    "wheel-build exit=0 elapsed=0s (maturin absent, skipped)" | Add-Content -Path $summaryFile
    "wheel-install exit=0 elapsed=0s (maturin absent, skipped)" | Add-Content -Path $summaryFile
    "pytest exit=0 elapsed=0s (maturin absent, skipped)" | Add-Content -Path $summaryFile
  } else {
    Invoke-GateStep "wheel-build" { maturin build --release --out wheels-gate }
    $wheel = Get-ChildItem -Path "wheels-gate" -Filter "graymul-*.whl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($wheel -eq $null) {
      "wheel-install exit=1 elapsed=0s (no wheel found)" | Add-Content -Path $summaryFile
      Write-Host "[gate] FAIL at wheel-install code 1 (no wheel found)"
      exit 1
    }
    $wheelPath = $wheel.FullName
    Invoke-GateStep "wheel-install" { pip install --force-reinstall --no-deps $wheelPath }
    Invoke-GateStep "pytest" { python -m pytest -q python/tests }
  }
  Invoke-GateStep "version-guard" { python scripts/check_version.py }
  Invoke-GateStep "api-surface" { python scripts/check_api_surface.py }
  Invoke-GateStep "invariants" { python scripts/check_invariants.py }
  Invoke-GateStep "perf-smoke" { python scripts/perf_smoke.py --cells tiny }
}

"gate PASS" | Add-Content -Path $summaryFile
Write-Host "[gate] PASS"
exit 0
