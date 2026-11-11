param(
    [string]$Python = "$PSScriptRoot\..\..\.venvs\rag\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$projectRag = (Resolve-Path -LiteralPath "$PSScriptRoot\..").Path

function Invoke-RecSysStep {
    param([string[]]$StepArguments)
    & $Python @StepArguments
    if ($LASTEXITCODE -ne 0) { throw "RecSys step failed with code $LASTEXITCODE" }
}

if (!(Test-Path -LiteralPath "$projectRag\artifacts\features\legacy\manifest.json")) {
    Invoke-RecSysStep -StepArguments @("-B", "$PSScriptRoot/../src/run.py", "build_recsys_features")
}
if (!(Test-Path -LiteralPath "$projectRag\artifacts\evaluation\results_recsys\experiment.json")) {
    Invoke-RecSysStep -StepArguments @("-B", "$PSScriptRoot/../src/run.py", "train_recsys")
}
Write-Host "Results: $projectRag\artifacts\evaluation\results_recsys\recsys_evaluation_report.md"
