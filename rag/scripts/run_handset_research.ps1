param([string]$Python = "$PSScriptRoot\..\..\.venvs\rag\Scripts\python.exe")
$ErrorActionPreference = "Stop"
$ragRoot = (Resolve-Path -LiteralPath "$PSScriptRoot\..").Path
$splitPath = "$ragRoot\data\external\benchmarks\handsets_only_v2"
$featureWork = "$ragRoot\artifacts\cache\handset_feature_work_v1"
$featurePath = "$ragRoot\artifacts\cache\handset_features_v1"

function Invoke-ResearchPython([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Research step failed; retain checkpoints and inspect output." }
}

if (-not (Test-Path -LiteralPath "$splitPath\manifest.json")) {
    Invoke-ResearchPython -Arguments @("-B", "$PSScriptRoot/../src/run.py", "prepare_handset_recsys")
}
if (-not (Test-Path -LiteralPath "$featureWork\images")) {
    Invoke-ResearchPython -Arguments @("-B", "$PSScriptRoot/../src/run.py", "seed_handset_images")
}
if (-not (Test-Path -LiteralPath "$featurePath\manifest.json")) {
    foreach ($phase in @("text", "clip", "finalize")) {
        Invoke-ResearchPython -Arguments @("-B", "$PSScriptRoot/../src/run.py", "build_connected_features", "--split", $splitPath,
            "--work", $featureWork, "--output", $featurePath, "--phase", $phase)
    }
}
$modelsPath = "$ragRoot\artifacts\models\recommendation\handsets_multimodal_v1"
$resultPath = "$ragRoot\artifacts\evaluation\results_handsets_multimodal_v1"
$lightModels = "$ragRoot\artifacts\models\recommendation\handsets_lightgcn_v1"
$lightResults = "$ragRoot\artifacts\evaluation\results_handsets_lightgcn_v1"
if (-not (Test-Path -LiteralPath "$lightResults\experiment.json")) {
    $lightArgs = @("-B", "$PSScriptRoot/../src/run.py", "train_recsys_portable", "--split", $splitPath,
        "--models", $lightModels, "--output", $lightResults, "--methods", "lightgcn", "--seeds", "42", "43", "44", "--epochs", "100")
    if (Test-Path -LiteralPath "$lightResults\run_plan.json") { $lightArgs += "--resume" }
    Invoke-ResearchPython -Arguments $lightArgs
}
Invoke-ResearchPython -Arguments @("-B", "$PSScriptRoot/../src/run.py", "verify_recsys_checkpoints", "--split", $splitPath,
    "--models", $lightModels, "--experiment", "$lightResults\experiment.json")
$trainArgs = @("-B", "$PSScriptRoot/../src/run.py", "train_recsys_portable", "--split", $splitPath, "--features", $featurePath,
    "--models", $modelsPath, "--output", $resultPath, "--methods", "vbpr", "bm3", "graph_text_only", "graph_multimodal", "graph_multimodal_gcl",
    "--seeds", "42", "43", "44", "--epochs", "100")
if (Test-Path -LiteralPath "$resultPath\run_plan.json") { $trainArgs += "--resume" }
Invoke-ResearchPython -Arguments $trainArgs
Invoke-ResearchPython -Arguments @("-B", "$PSScriptRoot/../src/run.py", "verify_recsys_checkpoints", "--split", $splitPath, "--features", $featurePath,
    "--models", $modelsPath, "--experiment", "$resultPath\experiment.json")
if (-not (Test-Path -LiteralPath "$ragRoot\artifacts\cache\handset_runtime_v1\manifest.json")) {
    Invoke-ResearchPython -Arguments @("-B", "$PSScriptRoot/../src/run.py", "prepare_connected_runtime", "--split", $splitPath, "--features", $featurePath,
        "--images", "$featureWork\images", "--output", "$ragRoot\artifacts\cache\handset_runtime_v1")
}
