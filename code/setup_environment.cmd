@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 goto failed

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: The project Python environment is missing.
    goto failed
)
if not exist ".environment\tmp" mkdir ".environment\tmp"
if not exist ".environment\pip-cache" mkdir ".environment\pip-cache"
set "TEMP=%CD%\.environment\tmp"
set "TMP=%TEMP%"
set "PIP_CACHE_DIR=%CD%\.environment\pip-cache"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_REQUIRE_VIRTUALENV=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "OMP_NUM_THREADS=4"
set "OPENBLAS_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"

".venv\Scripts\python.exe" -B -c "import sys; assert sys.version_info[:2] == (3,12), 'Python 3.12 is required'; print(sys.version)"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -B -m pip install --only-binary=:all: --index-url https://pypi.org/simple --retries 1 --timeout 15 -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -B -m pip check
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -B -c "import numpy; assert numpy.__version__ == '2.5.3'; print('NumPy:', numpy.__version__)"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -B check_data.py
if errorlevel 1 goto failed

echo.
echo SUCCESS: MP2 dependencies installed and official data check passed.
popd
pause
exit /b 0

:failed
echo.
echo SETUP FAILED. Please send the error shown above.
popd
pause
exit /b 1
