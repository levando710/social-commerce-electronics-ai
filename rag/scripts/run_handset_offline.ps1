param(
    [string]$Python = "$PSScriptRoot\..\..\.venvs\rag\Scripts\python.exe",
    [int]$Port = 7860,
    [switch]$Neural
)
$ErrorActionPreference = "Stop"
$ragRoot = (Resolve-Path -LiteralPath "$PSScriptRoot\..").Path
$retrievalMethod = if ($Neural) { "multimodal" } else { "legacy" }
& $Python -B "$PSScriptRoot/../src/run.py" "chat_app" --handsets --offline --method $retrievalMethod --port $Port --interaction-db "$ragRoot\artifacts\cache\handset_browser.sqlite"
if ($LASTEXITCODE -ne 0) { throw "Offline handset chat stopped with an error; inspect its output." }
