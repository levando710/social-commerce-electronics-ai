[CmdletBinding()]
param(
    [ValidateSet('Fetch', 'Prepare', 'Features', 'Train', 'Verify', 'Analyze', 'All')]
    [string]$Stage = 'All',
    [string]$Python = "$PSScriptRoot/../../.venvs/rag/Scripts/python.exe",
    [int]$EncoderBatchSize = 4,
    [int]$CpuThreadCount = 2,
    [int]$TrainingEpochs = 100,
    [int]$TrainingDimension = 16,
    [int]$TrainingBatchSize = 1024
)
# Text-only RecSys benchmark on MIND-small (user-sampled, leave-last-out on clicks).
# Fetch needs a Hugging Face read token (HF_TOKEN in rag/.env) from an account that accepted the dataset terms.
# Only methods that do not need images are trained: news items have no pictures or specifications.
$ErrorActionPreference = 'Stop'
$ragRoot = (Resolve-Path -LiteralPath "$PSScriptRoot/..").Path
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$source = "$ragRoot/data/external/mind_small"
$split = "$ragRoot/data/external/benchmarks/mind_small_text_v1"
$work = "$ragRoot/artifacts/cache/mind_small_work_v1"
$features = "$ragRoot/artifacts/cache/mind_small_features_v1"
$runName = 'mind_small_text_v1'
$models = "$ragRoot/artifacts/models/recommendation/$runName"
$results = "$ragRoot/artifacts/evaluation/results_$runName"
$analysis = "$ragRoot/artifacts/evaluation/results_mind_small_text_analysis_v1"

function Invoke-ResearchPython {
    param([string[]]$CommandArguments)
    & $Python -B @CommandArguments
    if ($LASTEXITCODE -ne 0) { throw "Research step failed: $($CommandArguments[0])" }
}

if ($Stage -in @('Fetch', 'All')) {
    Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", 'fetch_mind_small', '--output', $source)
}
if ($Stage -in @('Prepare', 'All') -and -not (Test-Path -LiteralPath "$split/manifest.json")) {
    Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", 'prepare_mind_recsys',
        '--source', "$source/MINDsmall_train", '--output', $split)
}
if ($Stage -in @('Features', 'All') -and -not (Test-Path -LiteralPath "$features/manifest.json")) {
    # Each phase checkpoints per item; a Windows PermissionError on a temporary file can stop it, so rerun to resume.
    foreach ($phase in @('images', 'text', 'clip', 'finalize')) {
        Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", 'build_connected_features', '--split', $split,
            '--work', $work, '--output', $features, '--phase', $phase,
            '--batch-size', "$EncoderBatchSize", '--threads', "$CpuThreadCount")
    }
}
if ($Stage -in @('Train', 'All')) {
    $training = @{
        Python = $Python; Split = $split; Features = $features; RunName = $runName; Device = 'cpu';
        Methods = @('lightgcn', 'graph_text_only', 'bm3');
        Seeds = @(42, 43, 44); Epochs = $TrainingEpochs; Dim = $TrainingDimension; BatchSize = $TrainingBatchSize
    }
    if (Test-Path -LiteralPath "$results/run_plan.json") { $training.Resume = $true }
    & "$PSScriptRoot/run_portable_training.ps1" @training
}
if ($Stage -in @('Verify', 'All')) {
    Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", 'verify_recsys_checkpoints', '--split', $split,
        '--features', $features, '--models', $models, '--experiment', "$results/experiment.json")
}
if ($Stage -in @('Analyze', 'All')) {
    Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", 'analyze_category_recsys',
        '--experiment', "$results/experiment.json", '--output', $analysis,
        '--pairs', 'lightgcn:graph_text_only,lightgcn:bm3')
}
