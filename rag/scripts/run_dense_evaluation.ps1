param(
    [string]$Python = "$PSScriptRoot\..\..\.venvs\rag\Scripts\python.exe",
    [string]$DenseIndex = "$PSScriptRoot\..\artifacts\runtime\catalog\dense_index",
    [string]$OutputDir = "$PSScriptRoot\..\artifacts\evaluation\results_dense",
    [switch]$DownloadModels
)

$ErrorActionPreference = "Stop"

function Invoke-ResearchStep {
    param([string[]]$StepArguments)
    & $Python @StepArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Research pipeline failed with exit code $LASTEXITCODE"
    }
}

if (!(Test-Path -LiteralPath "$DenseIndex\manifest.json")) {
    $buildArguments = @("-B", "$PSScriptRoot/../src/run.py", "build_dense_index", "--output", $DenseIndex)
    if ($DownloadModels) { $buildArguments += "--allow-download" }
    Invoke-ResearchStep -StepArguments $buildArguments
}

Invoke-ResearchStep -StepArguments @("-B", "$PSScriptRoot/../src/run.py", "evaluate_retrieval", "--dense-index", $DenseIndex, "--output-dir", $OutputDir)
Write-Host "Reports: $OutputDir"
