param(
    [string]$PythonExe = 'python',
    [string]$VenvDir = '.venv',
    [string]$Requirements = 'requirements.txt'
)

Write-Host "Using Python: $PythonExe"

try {
    & $PythonExe -V
} catch {
    Write-Error "Python not found at '$PythonExe'. Provide full path to python.exe or ensure python is in PATH."; exit 1
}

Write-Host "Creating virtualenv in '$VenvDir'..."
& $PythonExe -m venv $VenvDir

Write-Host "Enabling script execution for this session..."
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

Write-Host "Activate the virtualenv in this session now..."
. "$VenvDir\Scripts\Activate.ps1"

Write-Host "Upgrading pip and installing requirements (if present)..."
python -m pip install --upgrade pip
if (Test-Path $Requirements) {
    python -m pip install -r $Requirements
} else {
    Write-Host "No requirements file found at '$Requirements' - skipping pip install.";
}

Write-Host "Done. To reactivate the venv in a new shell run:`n .\$VenvDir\Scripts\Activate.ps1`"
