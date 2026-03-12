Param(
    [string]$PythonExe = "py",
    [string]$Config = "config/default.yaml",
    [switch]$SkipExternal,
    [switch]$SkipBenchmark
)

Write-Host "=== ECG Drift Guard Pipeline ===" -ForegroundColor Cyan
Write-Host "Python : $PythonExe"
Write-Host "Config : $Config"

function Run-Step {
    param(
        [string]$Title,
        [string]$Cmd
    )
    Write-Host ""
    Write-Host ">>> $Title" -ForegroundColor Yellow
    Write-Host "    $Cmd"
    & powershell -NoLogo -NoProfile -Command $Cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Step failed: $Title (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# 1) Splits
Run-Step "01_make_splits (DS1/DS2 + leakage check)" `
    "$PythonExe scripts/01_make_splits.py --config `"$Config`""

# 2) Build NPZ dataset
Run-Step "02_build_dataset (WFDB download + beat cut + NPZ)" `
    "$PythonExe scripts/02_build_dataset.py --config `"$Config`""

# 3) Train model
Run-Step "03_train_model (1D-CNN + embedder + bootstrap CI)" `
    "$PythonExe scripts/03_train_model.py --config `"$Config`""

# 4) Drift evaluation
Run-Step "04_drift_evaluate (scenarios × intensities + correlation)" `
    "$PythonExe scripts/04_drift_evaluate.py --config `"$Config`""

# 5) Calibration + Risk + Audit
Run-Step "05_calibrate_and_risk (calibration + risk policy + summary.json)" `
    "$PythonExe scripts/05_calibrate_and_risk.py --config `"$Config`""

# 6) Hypothesis report (optional but fast)
Run-Step "06_hypothesis_report (H1~H3 textual summary)" `
    "$PythonExe scripts/06_hypothesis_report.py --config `"$Config`""

if (-not $SkipExternal) {
    Run-Step "07_external_validation (external DB: svdb)" `
        "$PythonExe scripts/07_external_validation.py --config `"$Config`" --ext-db svdb"
} else {
    Write-Host "Skip 07_external_validation (per flag)" -ForegroundColor DarkYellow
}

Run-Step "08_model_registry_update (append current version)" `
    "$PythonExe scripts/08_model_registry_update.py --config `"$Config`""

if (-not $SkipBenchmark) {
    Run-Step "09_benchmark (latency + memory + device)" `
        "$PythonExe scripts/09_benchmark.py --config `"$Config`""
} else {
    Write-Host "Skip 09_benchmark (per flag)" -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "=== Pipeline finished successfully ===" -ForegroundColor Green
Write-Host "Key outputs:"
Write-Host "  artifacts/reports/summary.json"
Write-Host "  artifacts/reports/decisions.csv"
Write-Host "  artifacts/reports/figures/"

