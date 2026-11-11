param(
    [string]$Python = "$PSScriptRoot\..\..\.venvs\rag\Scripts\python.exe",
    [string]$RunName = "connected_multimodal_v2",
    [string]$Split = "$PSScriptRoot\..\data\external\benchmarks\connected_phones_v1",
    [string]$Features = "$PSScriptRoot\..\artifacts\cache\connected_features_v1",
    [ValidateSet("cpu", "cuda", "auto")][string]$Device = "cpu",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
if (!(Test-Path -LiteralPath "$Features\manifest.json")) {
    throw "Frozen feature store missing. Run build_connected_features.py phases images, text, clip, finalize first, or transfer the prepared multimodal bundle."
}
$parameters = @{
    Python = $Python; RunName = $RunName; Split = $Split; Features = $Features; Device = $Device;
    Methods = @("lightgcn", "vbpr", "bm3", "graph_text_only", "graph_multimodal", "graph_multimodal_gcl");
    Seeds = @(42, 43, 44); Epochs = 100; EvalEvery = 10; Dim = 16; BatchSize = 1024
}
if ($Resume) { $parameters.Resume = $true }
& "$PSScriptRoot\run_portable_training.ps1" @parameters
$ragRoot = (Resolve-Path -LiteralPath "$PSScriptRoot\..").Path
if (Test-Path -LiteralPath "$ragRoot\artifacts\evaluation\results_$RunName\experiment.json") {
    & $Python -B "$PSScriptRoot/../src/run.py" "verify_recsys_checkpoints" --split $Split --features $Features --models "$ragRoot\artifacts\models\recommendation\$RunName" --experiment "$ragRoot\artifacts\evaluation\results_$RunName\experiment.json"
    if ($LASTEXITCODE -ne 0) { throw "Checkpoint reconstruction failed; do not distribute this run" }
}
