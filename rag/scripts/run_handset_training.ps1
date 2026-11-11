param(
    [string]$Python = "$PSScriptRoot\..\..\.venvs\rag\Scripts\python.exe",
    [string]$RunName = "handsets_multimodal_v1",
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
$ragRoot = (Resolve-Path -LiteralPath "$PSScriptRoot\..").Path
$splitPath = "$ragRoot\data\external\benchmarks\handsets_only_v2"
$featurePath = "$ragRoot\artifacts\cache\handset_features_v1"
$parameters = @{
    Python = $Python; RunName = $RunName; Split = $splitPath; Features = $featurePath; Device = "cpu";
    Methods = @("vbpr", "bm3", "graph_text_only", "graph_multimodal", "graph_multimodal_gcl");
    Seeds = @(42, 43, 44); Epochs = 100; EvalEvery = 10; Dim = 16; BatchSize = 1024
}
if ($Resume) { $parameters.Resume = $true }
& "$PSScriptRoot\run_portable_training.ps1" @parameters
& $Python -B "$PSScriptRoot/../src/run.py" "verify_recsys_checkpoints" --split $splitPath --features $featurePath --models "$ragRoot\artifacts\models\recommendation\$RunName" --experiment "$ragRoot\artifacts\evaluation\results_$RunName\experiment.json"
if ($LASTEXITCODE -ne 0) { throw "Handset checkpoint reconstruction failed" }
