$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$envFile = Join-Path $projectRoot ".env"
if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Missing .env file at $envFile"
}

$keyLine = Get-Content -LiteralPath $envFile | Where-Object {
    $_ -match '^\s*GOOGLE_SAFE_BROWSING_API_KEY\s*='
} | Select-Object -Last 1

if (-not $keyLine) {
    throw "Add GOOGLE_SAFE_BROWSING_API_KEY=your_key to the .env file."
}

$apiKey = ($keyLine -replace '^\s*GOOGLE_SAFE_BROWSING_API_KEY\s*=\s*', '').Trim().Trim('"')
if ([string]::IsNullOrWhiteSpace($apiKey)) {
    throw "The Google Safe Browsing API key is empty in .env."
}

$env:GOOGLE_SAFE_BROWSING_API_KEY = $apiKey
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python environment not found at $python"
}

& $python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8001
