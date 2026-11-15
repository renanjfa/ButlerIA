$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if ((Test-Path -LiteralPath '.deps\sklearn') -and (Test-Path -LiteralPath $bundledPython)) {
    & $bundledPython app.py
} elseif (Test-Path -LiteralPath '.venv\Scripts\python.exe') {
    & '.\.venv\Scripts\python.exe' app.py
} else {
    python app.py
}
