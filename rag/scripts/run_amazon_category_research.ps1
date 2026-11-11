[CmdletBinding()]
param(
    [ValidateSet('Fetch', 'Prepare', 'Features', 'Train', 'Verify', 'Analyze', 'All')]
    [string]$Stage = 'All',
    # Amazon Reviews 2023 category and the short name used for run/cache folders (existing run: baby).
    [string]$Category = 'Baby_Products',
    [string]$Name = 'baby',
    [string]$Python = "$PSScriptRoot/../../.venvs/rag/Scripts/python.exe",
    [int]$EncoderBatchSize = 4,
    [int]$CpuThreadCount = 2,
    [int]$TrainingEpochs = 100,
    [int]$TrainingDimension = 16,
    [int]$TrainingBatchSize = 1024
)
# Reproduces the second RecSys benchmark (Amazon Reviews 2023, item-sampled 5-core, leave-last-out).
# Downloads public data and trains on CPU; sends nothing to any API.  Re-running a finished stage is safe:
# frozen outputs are never overwritten (Prepare/Train refuse or resume).
$ErrorActionPreference = 'Stop'
$ragRoot = (Resolve-Path -LiteralPath "$PSScriptRoot/..").Path
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$source = "$ragRoot/data/external/amazon_2023_$($Category.ToLower())"
$split = "$ragRoot/data/external/benchmarks/amazon2023_$($Category.ToLower())_v1"
$work = "$ragRoot/artifacts/cache/amazon2023_${Name}_work_v1"
$features = "$ragRoot/artifacts/cache/amazon2023_${Name}_features_v1"
$runName = "amazon2023_${Name}_v1"
$models = "$ragRoot/artifacts/models/recommendation/$runName"
$results = "$ragRoot/artifacts/evaluation/results_$runName"
$analysis = "$ragRoot/artifacts/evaluation/results_amazon2023_${Name}_analysis_v1"

function Invoke-ResearchPython {
    param([string[]]$CommandArguments)
    & $Python -B @CommandArguments
    if ($LASTEXITCODE -ne 0) { throw "Research step failed: $($CommandArguments[0])" }
}

if ($Stage -in @('Fetch', 'All')) {
    Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", 'fetch_amazon_category_2023',
        '--category', $Category, '--output', $source)
}
if ($Stage -in @('Prepare', 'All') -and -not (Test-Path -LiteralPath "$split/manifest.json")) {
    Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", 'prepare_amazon_category_recsys',
        '--rating-csv', "$source/$Category.csv", '--meta', "$source/meta_$Category.jsonl", '--output', $split)
}
if ($Stage -in @('Features', 'All') -and -not (Test-Path -LiteralPath "$features/manifest.json")) {
    foreach ($phase in @('images', 'text', 'clip', 'finalize')) {
        Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", 'build_connected_features', '--split', $split,
            '--work', $work, '--output', $features, '--phase', $phase,
            '--batch-size', "$EncoderBatchSize", '--threads', "$CpuThreadCount")
    }
}
if ($Stage -in @('Train', 'All')) {
    $training = @{
        Python = $Python; Split = $split; Features = $features; RunName = $runName; Device = 'cpu';
        Methods = @('lightgcn', 'vbpr', 'bm3', 'graph_text_only', 'graph_multimodal', 'graph_multimodal_gcl');
        Seeds = @(42, 43, 44); Epochs = $TrainingEpochs; Dim = $TrainingDimension; BatchSize = $TrainingBatchSize
    }
    # A Windows PermissionError while renaming a temporary checkpoint can stop a long run;
    # resuming keeps the recorded source/config contract.
    if (Test-Path -LiteralPath "$results/run_plan.json") { $training.Resume = $true }
    & "$PSScriptRoot/run_portable_training.ps1" @training
}
if ($Stage -in @('Verify', 'All')) {
    Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", 'verify_recsys_checkpoints', '--split', $split,
        '--features', $features, '--models', $models, '--experiment', "$results/experiment.json")
}

if ($Stage -in @('Analyze', 'All')) {
    Invoke-ResearchPython -CommandArguments @("$PSScriptRoot/../src/run.py", 'analyze_category_recsys',
        '--experiment', "$results/experiment.json", '--output', $analysis)
}
