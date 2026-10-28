# Run this from a terminal with access to PyPI, or pass a local wheel directory.
param([string]$WheelDirectory)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'activate_project.ps1')

& $projectPython -B -c "import sys; assert sys.version_info[:2] == (3, 12), 'Python 3.12 is required'; print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw 'Python version verification failed.' }

$pipArguments = @('-B', '-m', 'pip', 'install', '--only-binary=:all:', '--retries', '1', '--timeout', '15')
if ($WheelDirectory) {
    $pipArguments += @('--no-index', '--find-links', (Resolve-Path -LiteralPath $WheelDirectory).Path)
} else {
    $pipArguments += @('--index-url', 'https://pypi.org/simple')
}
$pipArguments += @('-r', (Join-Path $PSScriptRoot 'requirements.txt'))
& $projectPython @pipArguments
if ($LASTEXITCODE -ne 0) {
    throw 'Dependency installation failed. Re-run this script in a terminal with network access, or supply -WheelDirectory with the official NumPy 2.5.3 wheel.'
}

& $projectPython -B -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
& $projectPython -B -c "import numpy; assert numpy.__version__ == '2.5.3', 'The official NumPy version is required'; print('NumPy:', numpy.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'NumPy version verification failed.' }
& $projectPython -B (Join-Path $PSScriptRoot 'check_data.py')
if ($LASTEXITCODE -ne 0) { throw 'The official data check failed.' }

Write-Host 'MP2 environment is ready; official data check passed.'
