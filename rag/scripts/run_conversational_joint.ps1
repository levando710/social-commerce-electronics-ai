param([string]$Python = "$PSScriptRoot\..\..\.venvs\rag\Scripts\python.exe", [switch]$Resume)
$ErrorActionPreference = "Stop"
$jointArgs = @("-B", "$PSScriptRoot/../src/run.py", "train_conversational_joint")
if ($Resume) { $jointArgs += "--resume" }
& $Python @jointArgs
if ($LASTEXITCODE -ne 0) { throw "Conversational training failed; inspect the saved checkpoint." }
$auditArgs = @("-B", "$PSScriptRoot/../src/run.py", "audit_conversational_results")
if (Test-Path -LiteralPath "$PSScriptRoot\..\artifacts\evaluation\results_conversational_joint_analysis_v1") {
    $auditOutput = "$PSScriptRoot\..\artifacts\evaluation\results_conversational_joint_audit_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')"
    $auditArgs += @("--output", $auditOutput)
}
& $Python @auditArgs
if ($LASTEXITCODE -ne 0) { throw "Checkpoint audit failed; do not report the run as verified." }
