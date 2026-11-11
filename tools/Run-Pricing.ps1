[CmdletBinding()]
param(
    [string]$Python,
    [int]$Port = 8501,
    [switch]$Headless
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pricingRoot = Join-Path $projectRoot 'pricing'
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $projectRoot '.venvs/pricing/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Pricing Python environment is missing. Follow pricing/README.md to install it.'
}
$Python = (Resolve-Path -LiteralPath $Python).Path
$headlessValue = $Headless.IsPresent.ToString().ToLowerInvariant()
Push-Location -LiteralPath $pricingRoot
try {
    & $Python -B -m streamlit run 'src/phone_pricing/apps/app_demo.py' --server.port $Port --server.headless $headlessValue --browser.gatherUsageStats false
    if ($LASTEXITCODE -ne 0) { throw 'Pricing application stopped with an error.' }
} finally {
    Pop-Location
}
