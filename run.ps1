# ============================================================
# PHISH//X - local launcher
# ============================================================
#
# Usage, from the project root:
#
#     .\run.ps1
#
# Requires the project virtual environment (see README > Setup).
# Optional server-side keys are read from the Git-ignored root
# .env file by webapp/app.py itself, so this script does not
# handle secrets.
# ============================================================

$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found at $python. See README.md > Setup."
}

& $python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000
