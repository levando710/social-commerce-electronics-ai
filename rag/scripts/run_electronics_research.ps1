param(
    [ValidateSet('Prepare', 'Features', 'Train', 'Retrieval', 'Coupling', 'Joint', 'All')]
    [string]$Stage = 'All',
    [string]$Python = "$PSScriptRoot/../../.venvs/rag/Scripts/python.exe",
    [int]$EncoderBatchSize = 4,
    [int]$CpuThreadCount = 2,
    [int]$TrainingEpochs = 100,
    [int]$MaximumSupportUsers = 2000,
    [int]$MinimumPositiveItems = 5,
    [int]$TrainingDimension = 16,
    [int]$TrainingBatchSize = 1024
)
$ErrorActionPreference = 'Stop'
$ragRoot = (Resolve-Path -LiteralPath "$PSScriptRoot/..").Path
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$sourceSplit = "$ragRoot/data/external/benchmarks/electronics_v1"
$draftSplit = "$ragRoot/data/external/benchmarks/connected_electronics_v1"
$split = "$ragRoot/data/external/benchmarks/connected_electronics_v2"
$archive = "$ragRoot/data/external/amazon_5core_electronics/Electronics.csv.gz"
$work = "$ragRoot/artifacts/cache/electronics_catalog_work_v1"
$catalogFeatures = "$ragRoot/artifacts/cache/electronics_catalog_features_v1"
$features = "$ragRoot/artifacts/cache/electronics_features_v1"
$queries = "$ragRoot/artifacts/cache/electronics_query_vectors_v1"
$runName = 'electronics_multimodal_v1'

function Invoke-ResearchPython {
    param([string[]]$CommandArguments)
    & $Python -B @CommandArguments
    if ($LASTEXITCODE -ne 0) { throw "Research step failed: $($CommandArguments[0])" }
}

if ($Stage -in @('Prepare', 'All')) {
    Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", "fetch_electronics_5core")
    if (-not (Test-Path -LiteralPath "$draftSplit/manifest.json")) {
        Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", "prepare_connected_recsys", '--base', $sourceSplit,
            '--source', $archive, '--output', $draftSplit, '--max-support-users', "$MaximumSupportUsers", '--min-support-items', "$MinimumPositiveItems")
    }
    if (-not (Test-Path -LiteralPath "$split/manifest.json")) {
        Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", "finalize_electronics_split", '--source', $draftSplit, '--output', $split)
    }
}
if ($Stage -in @('Features', 'All')) {
    if (-not (Test-Path -LiteralPath "$catalogFeatures/manifest.json")) {
        foreach ($phase in @('images', 'text', 'clip', 'finalize')) {
            Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", "build_connected_features", '--split', $sourceSplit,
                '--work', $work, '--output', $catalogFeatures, '--phase', $phase,
                '--batch-size', "$EncoderBatchSize", '--threads', "$CpuThreadCount")
        }
    }
    if (-not (Test-Path -LiteralPath "$features/manifest.json")) {
        Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", "rebind_catalog_features", '--source-split', $sourceSplit,
            '--source-features', $catalogFeatures, '--target-split', $split, '--output', $features)
    }
    Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", "audit_training_readiness", '--split', $split, '--base', $sourceSplit,
        '--features', $features, '--output', "$ragRoot/data/discovery/electronics_training_audit.json")
}
if ($Stage -in @('Train', 'All')) {
    $trainingParameters = @{
        Python = $Python; Split = $split; Features = $features; RunName = $runName; Device = 'cpu';
        Methods = @('lightgcn', 'vbpr', 'bm3', 'graph_text_only', 'graph_multimodal', 'graph_multimodal_gcl');
        Seeds = @(42, 43, 44); Epochs = $TrainingEpochs; Dim = $TrainingDimension; BatchSize = $TrainingBatchSize
    }
    if (Test-Path -LiteralPath "$ragRoot/artifacts/evaluation/results_$runName/run_plan.json") { $trainingParameters.Resume = $true }
    & "$PSScriptRoot/run_portable_training.ps1" @trainingParameters
    Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", "verify_recsys_checkpoints", '--split', $split,
        '--features', $features, '--models', "$ragRoot/artifacts/models/recommendation/$runName",
        '--experiment', "$ragRoot/artifacts/evaluation/results_$runName/experiment.json")
}
if ($Stage -in @('Retrieval', 'All')) {
    foreach ($phase in @('text-queries', 'clip-queries')) {
        Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", "evaluate_connected_retrieval", '--split', $split,
            '--features', $features, '--work', $queries, '--phase', $phase,
            '--batch-size', "$EncoderBatchSize", '--threads', "$CpuThreadCount")
    }
    if (-not (Test-Path -LiteralPath "$ragRoot/artifacts/evaluation/results_electronics_ir_v1/experiment.json")) {
        Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", "evaluate_connected_retrieval", '--split', $split,
            '--features', $features, '--work', $queries, '--image-dir', "$work/images", '--phase', 'evaluate',
            '--output', "$ragRoot/artifacts/evaluation/results_electronics_ir_v1")
    }
    if (-not (Test-Path -LiteralPath "$ragRoot/artifacts/cache/electronics_runtime_v1/manifest.json")) {
        Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", "prepare_connected_runtime", '--split', $split,
            '--features', $features, '--images', "$work/images", '--output', "$ragRoot/artifacts/cache/electronics_runtime_v1")
    }
}
if ($Stage -in @('Coupling', 'All')) {
    if (-not (Test-Path -LiteralPath "$ragRoot/artifacts/evaluation/results_electronics_coupling_v1/experiment.json")) {
        Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", "evaluate_electronics_coupling")
    }
}
if ($Stage -in @('Joint', 'All')) {
    $jointArguments = @("$PSScriptRoot/../src/run.py", "train_conversational_joint", '--split', $split,
        '--features', $features, '--queries', $queries, '--epochs', "$TrainingEpochs",
        '--output', "$ragRoot/artifacts/evaluation/results_electronics_joint_v1", '--models', "$ragRoot/artifacts/models/joint/electronics_v1")
    if (Test-Path -LiteralPath "$ragRoot/artifacts/evaluation/results_electronics_joint_v1/run_plan.json") { $jointArguments += '--resume' }
    Invoke-ResearchPython -CommandArguments $jointArguments
}
