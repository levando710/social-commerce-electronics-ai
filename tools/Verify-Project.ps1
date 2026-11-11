[CmdletBinding()]
param(
    [string]$Python = (Join-Path $PSScriptRoot '..\.venvs\rag\Scripts\python.exe')
)
# One command that re-checks everything that needs no network, no training and no API key:
# unit tests, project structure and frozen hashes, and every number in the report against its result files.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath "$PSScriptRoot/..").Path
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $root

function Invoke-Step {
    param([string]$Name, [string[]]$Arguments)
    Write-Host "== $Name"
    # unittest writes its summary to stderr; judge each step by its exit code, not by stderr text.
    $ErrorActionPreference = 'Continue'
    & $Python -B @Arguments
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code -ne 0) { throw "Step failed: $Name" }
}

Invoke-Step 'RAG tests' @('-m', 'unittest', 'discover', '-s', 'rag/tests', '-q')
Invoke-Step 'Repository tooling tests' @('-m', 'unittest', 'discover', '-s', 'tests', '-q')
Invoke-Step 'Project structure and frozen inputs' @('tools/check_project.py', '--require-evidence')
Invoke-Step 'Report numbers against result files' @('tools/check_report_numbers.py')
Write-Host 'All checks passed.'
