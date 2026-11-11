param(
    [string]$Python = "$PSScriptRoot\..\..\.venvs\rag\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/4] Prepare multimodal-aware corpus"
& $Python src/run.py prepare_corpus

Write-Host "[2/4] Build BM25 index"
& $Python src/run.py build_index

Write-Host "[3/4] Audit multimodal coverage"
& $Python src/run.py audit_multimodal_coverage

Write-Host "[4/4] Evaluate retrieval baselines"
& $Python src/run.py evaluate_retrieval

Write-Host ""
Write-Host "Done. Reports:"
Write-Host "- evaluation\results\multimodal_coverage.json"
Write-Host "- evaluation\results\retrieval_evaluation_report.md"
Write-Host "- evaluation\results\retrieval_metrics_by_query.csv"
Write-Host "- evaluation\results\retrieval_top_results.csv"
