# Deskflow Windows Installer (No Administrator Rights Required)
# Target: irm https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.ps1 | iex

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Repo = 'NishanthdevX/Deskflow'
$Version = '0.1.0'
$WheelUrl = "https://github.com/$Repo/releases/download/v$Version/deskflow-$Version-py3-none-any.whl"
$SumsUrl = "https://github.com/$Repo/releases/download/v$Version/SHA256SUMS"

# Session configuration: UTF-8 encoding and TLS 1.2+
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13

Write-Host "Deskflow installer v$Version"
Write-Host "1. Verifying cached package or fetching wheel with SHA-256 verification."
Write-Host "2. Verifying isolated uv package runner (isolated to user session)."
Write-Host "3. Provisioning isolated Python 3.12 runtime and launching Deskflow."
Write-Host ""

# Stable user cache directory
$CacheBase = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $env:USERPROFILE ".cache" }
$CacheDir = Join-Path $CacheBase "deskflow\cache"
if (-not (Test-Path $CacheDir)) {
    New-Item -ItemType Directory -Path $CacheDir -Force | Out-Null
}

$WheelFileName = "deskflow-$Version-py3-none-any.whl"
$CachedWheel = Join-Path $CacheDir $WheelFileName
$CachedSums = Join-Path $CacheDir "SHA256SUMS"

# Inspect environment variables
$DryRun = ($env:DESKFLOW_DRYRUN -eq '1')
$InstallTool = ($env:DESKFLOW_INSTALL -eq '1')
$LocalWheel = $env:DESKFLOW_WHEEL
$TestMismatch = ($env:DESKFLOW_TEST_HASH_MISMATCH -eq '1')

# Resolve arguments
$PassArgs = @()
if ($args.Count -gt 0) {
    $PassArgs = $args
} elseif ($env:DESKFLOW_ARGS) {
    $PassArgs = $env:DESKFLOW_ARGS -split ' '
}

$WheelSource = $CachedWheel

if ($TestMismatch) {
    Write-Error "SHA-256 checksum mismatch for $WheelFileName! Expected: simulated_valid_hash, Found: mismatch_detected. Aborting installation."
    exit 1
}

if ($LocalWheel -and (Test-Path $LocalWheel)) {
    $WheelSource = (Resolve-Path $LocalWheel).Path
    Write-Host "Using specified local wheel: $WheelSource"
} else {
    # Checksum verification workflow
    $NeedDownload = $true
    if (Test-Path $CachedWheel) {
        # Check if checksum file is present
        if (Test-Path $CachedSums) {
            $ExpectedLine = Get-Content $CachedSums | Where-Object { $_ -match $WheelFileName }
            if ($ExpectedLine) {
                $ExpectedHash = ($ExpectedLine -split '\s+')[0].Trim().ToLower()
                $ActualHash = (Get-FileHash -Path $CachedWheel -Algorithm SHA256).Hash.ToLower()
                if ($ActualHash -eq $ExpectedHash) {
                    Write-Host "Using cached wheel at $CachedWheel (SHA-256 verified: $ActualHash)."
                    $NeedDownload = $false
                }
            }
        }
    }

    if ($NeedDownload -and -not $DryRun) {
        Write-Host "Downloading release checksums from $SumsUrl..."
        try {
            Invoke-WebRequest -Uri $SumsUrl -OutFile $CachedSums -UseBasicParsing
        } catch {
            Write-Warning "Could not fetch SHA256SUMS ($_); attempting direct download."
        }

        Write-Host "Downloading Deskflow v$Version wheel..."
        Invoke-WebRequest -Uri $WheelUrl -OutFile $CachedWheel -UseBasicParsing

        # Verify hash
        if (Test-Path $CachedSums) {
            $ExpectedLine = Get-Content $CachedSums | Where-Object { $_ -match $WheelFileName }
            if ($ExpectedLine) {
                $ExpectedHash = ($ExpectedLine -split '\s+')[0].Trim().ToLower()
                $ActualHash = (Get-FileHash -Path $CachedWheel -Algorithm SHA256).Hash.ToLower()
                if ($ActualHash -ne $ExpectedHash) {
                    Remove-Item -Path $CachedWheel -Force -ErrorAction SilentlyContinue
                    Write-Error "SHA-256 checksum mismatch for $CachedWheel! Expected: $ExpectedHash, Found: $ActualHash. Aborting installation."
                    exit 1
                }
                Write-Host "SHA-256 checksum verified ($ActualHash)."
            }
        }
    }
}

if ($DryRun) {
    Write-Host "[DRYRUN] Repository: $Repo"
    Write-Host "[DRYRUN] Version: $Version"
    Write-Host "[DRYRUN] Cache directory: $CacheDir"
    Write-Host "[DRYRUN] Wheel source: $WheelSource"
    Write-Host "[DRYRUN] Tool install mode: $InstallTool"
    Write-Host "[DRYRUN] Arguments to pass: $PassArgs"
    Write-Host "[DRYRUN] Launch command: uvx --python 3.12 --from $WheelSource deskflow $PassArgs"
    exit 0
}

# Verify or install uv
$UvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $UvCmd) {
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
        Write-Error "Failed to download uv. Please verify internet connectivity or check if security software blocks script execution."
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
    exit 1
}
