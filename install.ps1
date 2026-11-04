# Deskflow Windows Installer (No Administrator Rights Required)
# Target: irm https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.ps1 | iex

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Repo = 'NishanthdevX/Deskflow'
$Version = '0.1.0'
$WheelUrl = "https://github.com/$Repo/releases/download/v$Version/deskflow-$Version-py3-none-any.whl"

# Session configuration: UTF-8 encoding and TLS 1.2+
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13

Write-Host "Deskflow installer v$Version"
Write-Host "1. Verifying or fetching uv package runner (isolated to user session)."
Write-Host "2. Provisioning isolated Python 3.12 runtime and dependencies on first launch."
Write-Host "3. Launching Deskflow. No administrator rights or permanent system changes are required."
Write-Host ""

# Inspect environment variables
$DryRun = ($env:DESKFLOW_DRYRUN -eq '1')
$InstallTool = ($env:DESKFLOW_INSTALL -eq '1')
$LocalWheel = $env:DESKFLOW_WHEEL

$WheelSource = $WheelUrl
if ($LocalWheel -and (Test-Path $LocalWheel)) {
    $WheelSource = (Resolve-Path $LocalWheel).Path
}

# Resolve arguments
$PassArgs = @()
if ($args.Count -gt 0) {
    $PassArgs = $args
} elseif ($env:DESKFLOW_ARGS) {
    $PassArgs = $env:DESKFLOW_ARGS -split ' '
}

if ($DryRun) {
    Write-Host "[DRYRUN] Repository: $Repo"
    Write-Host "[DRYRUN] Version: $Version"
    Write-Host "[DRYRUN] Wheel source: $WheelSource"
    Write-Host "[DRYRUN] Tool install mode: $InstallTool"
    Write-Host "[DRYRUN] Arguments to pass: $PassArgs"
    Write-Host "[DRYRUN] UV check: checking Get-Command uv"
    Write-Host "[DRYRUN] Launch command: uvx --python 3.12 --from $WheelSource deskflow $PassArgs"
    exit 0
}

# Verify or install uv
$UvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $UvCmd) {
    # Check default user bin directory for uv if installed previously
    $UserUvBin = Join-Path $env:USERPROFILE ".local\bin"
    $UserUvExe = Join-Path $UserUvBin "uv.exe"
    if (Test-Path $UserUvExe) {
        $env:PATH = "$UserUvBin;$env:PATH"
        $UvCmd = Get-Command uv -ErrorAction SilentlyContinue
    }
}

if (-not $UvCmd) {
    Write-Host "Fetching isolated uv package manager..."
    $env:UV_NO_MODIFY_PATH = '1'
    try {
        & ([scriptblock]::Create((irm https://astral.sh/uv/install.ps1)))
    } catch {
        Write-Error "Failed to download uv. Please verify internet connectivity or check if your security software blocks script execution. Fallback: download install.ps1, inspect it, and run locally."
        exit 1
    }

    $UserUvBin = Join-Path $env:USERPROFILE ".local\bin"
    if (Test-Path (Join-Path $UserUvBin "uv.exe")) {
        $env:PATH = "$UserUvBin;$env:PATH"
    } else {
        $CargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
        if (Test-Path (Join-Path $CargoBin "uv.exe")) {
            $env:PATH = "$CargoBin;$env:PATH"
        } else {
            Write-Error "uv was installed but the executable could not be located in session PATH."
            exit 1
        }
    }
}

# Run or install Deskflow
try {
    if ($InstallTool) {
        Write-Host "Installing Deskflow to user tool environment..."
        & uv tool install --python 3.12 --from $WheelSource deskflow --force
        Write-Host "Deskflow installed. Run 'deskflow' from your terminal."
    } else {
        if ($PassArgs.Count -gt 0) {
            & uvx --python 3.12 --from $WheelSource deskflow @PassArgs
        } else {
            & uvx --python 3.12 --from $WheelSource deskflow
        }
    }
} catch {
    Write-Error "Deskflow launch failed: $_"
    Write-Host "If security software interrupted the command, download the wheel and install locally with: uvx --from <wheel> deskflow"
    exit 1
}

