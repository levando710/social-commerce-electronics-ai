param(
    [int]$MaxProducts = 1000,
    [int]$ReviewsPerProduct = 20,
    [int]$ScanLimit = 1000000,
    [int]$ImagesPerRecord = 2,
    [int]$MaxImages = 2000,
    [string]$Output = "data/collections/phones_v1"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root "..\.venvs\rag\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Không tìm thấy Python venv tại $Python. Hãy chạy: py -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

Set-Location $Root

& $Python src/run.py collect_amazon metadata `
    --output $Output `
    --max-products $MaxProducts `
    --scan-limit $ScanLimit

& $Python src/run.py collect_amazon reviews `
    --output $Output `
    --reviews-per-product $ReviewsPerProduct `
    --scan-limit $ScanLimit

& $Python src/run.py collect_amazon images `
    --output $Output `
    --images-per-record $ImagesPerRecord `
    --max-images $MaxImages

& $Python src/run.py collect_amazon export --output $Output

Write-Host ""
Write-Host "Đã hoàn tất một lượt cào/xuất dữ liệu."
Write-Host "Dữ liệu nằm trong: $Output\processed"
Write-Host "Nếu summary.json báo eof=false và chưa đủ review, chạy lại đúng lệnh này để tiếp tục."
