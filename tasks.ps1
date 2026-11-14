param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('test', 'lint', 'run-app')]
    [string] $Task
)

$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found at $python. See README.md > Setup."
}

Set-Location -LiteralPath $projectRoot

switch ($Task) {
    'test' {
        & $python -m pytest -q
        exit $LASTEXITCODE
    }
    'lint' {
        & $python -m ruff check src webapp scripts tests
        exit $LASTEXITCODE
    }
    'run-app' {
        & $python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000
        exit $LASTEXITCODE
    }
}
