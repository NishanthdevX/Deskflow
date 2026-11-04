# Deskflow Installation and Verification Guide

Deskflow requires no administrator or root privileges and installs in an isolated user-level environment.

---

## 1. Quick One-Liner Installation

### Windows (PowerShell 5.1+ or PowerShell 7+)
Run from Windows Terminal or PowerShell:
```powershell
irm https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.ps1 | iex
```

### macOS (Terminal)
Run from Terminal:
```bash
curl -fsSL https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.sh | bash
```

Both installers automatically:
1. Check for `uv` (fast Python package runner) and download it to your user directory if missing.
2. Download the Deskflow distribution wheel and official release `SHA256SUMS`.
3. Verify the SHA-256 cryptographic hash of the wheel before running.
4. Launch Deskflow directly using an isolated Python 3.12 virtual environment.

---

## 2. Checksum Verification

### Automatic Verification
The installer script downloads `SHA256SUMS` alongside the wheel. If the computed SHA-256 checksum does not match the published release hash, the installation immediately aborts:
```text
Error: SHA-256 checksum mismatch for deskflow-0.1.0-py3-none-any.whl!
Expected: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
Found:    a5892c17...
Aborting installation.
```

### Manual Verification
To manually inspect and verify release integrity:
1. Download the release wheel and `SHA256SUMS` from the GitHub Release.
2. Verify with PowerShell:
   ```powershell
   Get-FileHash -Path .\deskflow-0.1.0-py3-none-any.whl -Algorithm SHA256
   Get-Content .\SHA256SUMS
   ```
3. Or on macOS / Linux:
   ```bash
   shasum -a 256 -c SHA256SUMS
   ```

---

## 3. Persistent Tool Installation

To install `deskflow` permanently into your user command path (so you can run `deskflow` directly from any terminal window):

```powershell
# Windows
$env:DESKFLOW_INSTALL = "1"
irm https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.ps1 | iex
```

```bash
# macOS
export DESKFLOW_INSTALL=1
curl -fsSL https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.sh | bash
```

---

## 4. Uninstallation and Data Removal

### Remove Executable
If installed via `uv tool`:
```bash
uv tool uninstall deskflow
```

### Remove All Local Data
To permanently purge all local hardware baselines, telemetry traces, and fix audit logs:
```bash
deskflow uninstall --data
```
Deskflow will request typed confirmation (`delete my data`) before removing the local data directory.

---

## 5. Troubleshooting & Best Practices

### Windows Terminal Recommendation
For the optimal terminal experience, run Deskflow inside modern **Windows Terminal** rather than legacy Windows Console (`conhost.exe`). Windows Terminal includes native support for:
- 24-bit ANSI color and smooth gradients
- Unicode box drawings, block shading, and braille metric sparklines
- Smooth font rendering with Cascadia Code or modern monospace fonts

If running inside legacy Command Prompt or non-UTF-8 consoles, pass `--plain` for ASCII-safe fallback rendering:
```bash
deskflow --plain
```

### Execution Policy Restrictions (Windows)
If PowerShell blocks script execution:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Offline / Air-Gapped Installation
To install from a pre-downloaded wheel:
```bash
# Windows
$env:DESKFLOW_WHEEL = "C:\path\to\deskflow-0.1.0-py3-none-any.whl"
powershell -File install.ps1

# macOS
export DESKFLOW_WHEEL="/path/to/deskflow-0.1.0-py3-none-any.whl"
bash install.sh
```
