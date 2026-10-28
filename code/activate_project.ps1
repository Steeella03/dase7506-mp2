# Dot-source this file to use the MP2 environment in the current PowerShell.
$projectCodeDirectory = $PSScriptRoot
$projectPython = Join-Path $projectCodeDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'The project virtual environment is missing.'
}

$projectEnvironmentDirectory = Join-Path $projectCodeDirectory '.environment'
$projectTemporaryDirectory = Join-Path $projectEnvironmentDirectory 'tmp'
$projectPipCacheDirectory = Join-Path $projectEnvironmentDirectory 'pip-cache'
New-Item -ItemType Directory -Path $projectTemporaryDirectory,$projectPipCacheDirectory -Force | Out-Null

$env:TEMP = $projectTemporaryDirectory
$env:TMP = $projectTemporaryDirectory
$env:PIP_CACHE_DIR = $projectPipCacheDirectory
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:PIP_REQUIRE_VIRTUALENV = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'

. (Join-Path $projectCodeDirectory '.venv\Scripts\Activate.ps1')
Set-Location -LiteralPath $projectCodeDirectory
Write-Host 'MP2 virtual environment activated. Project caches and temporary files use D:.'
