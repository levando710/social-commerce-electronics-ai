param(
    [string]$Python = "$PSScriptRoot\..\..\.venvs\rag\Scripts\python.exe",
    [string]$Split = "$PSScriptRoot\..\data\external\benchmarks\connected_phones_v1",
    [string]$Features = "",
    [string]$RunName = "connected_phones_cpu_v1",
    [string[]]$Methods = @("lightgcn"),
    [int[]]$Seeds = @(42, 43, 44),
    [int]$Epochs = 100,
    [int]$EvalEvery = 10,
    [int]$Dim = 16,
    [int]$BatchSize = 1024,
    [ValidateSet("cpu", "cuda", "auto")][string]$Device = "cpu",
    [int]$MaxEpochsThisInvocation = 0,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw "RunName must be a single directory name" }
$ragRoot = (Resolve-Path -LiteralPath "$PSScriptRoot\..").Path
$arguments = @("-B", "$PSScriptRoot/../src/run.py", "train_recsys_portable", "--split", $Split,
    "--models", "$ragRoot\artifacts\models\recommendation\$RunName", "--output", "$ragRoot\artifacts\evaluation\results_$RunName",
    "--methods") + $Methods + @("--seeds") + @($Seeds | ForEach-Object { $_.ToString() }) + @(
    "--epochs", "$Epochs", "--eval-every", "$EvalEvery", "--dim", "$Dim", "--batch-size", "$BatchSize",
    "--threads", "2", "--device", $Device, "--max-epochs-this-invocation", "$MaxEpochsThisInvocation")
if ($Features) { $arguments += @("--features", $Features) }
if ($Resume) { $arguments += "--resume" }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Training stopped with exit code $LASTEXITCODE. Last completed epoch remains saved." }
Write-Host "Progress/checkpoints: $ragRoot\artifacts\models\recommendation\$RunName"
Write-Host "Results (after all runs complete): $ragRoot\artifacts\evaluation\results_$RunName"
