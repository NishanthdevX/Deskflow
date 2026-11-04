# Deskflow Installation and Packaging Guide

This document details the one-line installer mechanisms, downloaded payload sizes, security guarantees, baseline storage, and uninstallation instructions.

---

## 1. Quick Start Installers

### Windows PowerShell (No Administrator Rights Required)
```powershell
irm https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.ps1 | iex
```

### macOS Terminal (No Sudo Required)
```bash
curl -fsSL https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.sh | bash
```

> **Note on macOS**: The macOS installer is verified in continuous integration (CI) on `macos-latest` runners.

---

## 2. How the Installer Works and No-Admin Guarantee

The installation scripts (`install.ps1` and `install.sh`) are designed to be short, transparent, and completely non-destructive:

1. **No Administrator Rights**: The scripts run entirely in the user space and never request or require elevation (UAC / `sudo`).
2. **Session-Isolated PATH**: If the `uv` tool runner is not already installed on the machine, the installer invokes Astral's official user-space installer with `UV_NO_MODIFY_PATH=1` and appends `uv` to the current session `$env:PATH` only. It never edits system environment variables, Windows Registry, or shell RC profiles.
3. **Isolated Python Runtime**: Downloads an isolated, standalone Python 3.12 managed runtime inside the user's `~/.local/share/uv` or `%LOCALAPPDATA%\uv`.
4. **Execution via `uvx`**: Executes Deskflow inside an ephemeral environment directly from the packaged wheel:
   ```bash
   uvx --python 3.12 --from <wheel-url> deskflow
   ```

---

## 3. Measured Payload and Environment Sizes

Measurements conducted on version 0.1.0:

| Component | Measured Size | Notes |
| :--- | :--- | :--- |
| **Deskflow Wheel (`.whl`)** | ~1.23 MB | Includes source package and default pre-bundled models under 10 MB limit. |
| **Source Archive (`.tar.gz`)** | ~1.24 MB | Full source distribution. |
| **Dependencies & Runtime** | ~180 MB | Ephemeral virtual environment cached by `uv` (numpy, scikit-learn, xgboost, typer, rich). |
| **`xgboost` distribution** | ~55.5 MB | Standard CPU wheel (verified compatible on Windows and macOS; smaller than GPU variants). |

### Launch Latency Benchmark
- **First Launch (empty cache):** ~15 to 25 seconds (downloads Python 3.12 and dependencies via `uv`).
- **Subsequent Launches (cached):** < 1.2 seconds to reach the interactive home menu.

---

## 4. Environment Variables and Offline Modes

The installers support the following configuration flags:

- `DESKFLOW_ARGS`: Space-separated arguments passed to `deskflow` (e.g. `DESKFLOW_ARGS="doctor"` or `DESKFLOW_ARGS="scan"`).
- `DESKFLOW_WHEEL`: Path to a local wheel file for air-gapped or CI offline testing:
  ```powershell
  $env:DESKFLOW_WHEEL = "dist\deskflow-0.1.0-py3-none-any.whl"
  powershell -File install.ps1
  ```
- `DESKFLOW_INSTALL=1`: Performs a persistent user tool installation via `uv tool install` rather than transient execution:
  ```powershell
  $env:DESKFLOW_INSTALL = "1"
  irm https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.ps1 | iex
  ```
- `DESKFLOW_DRYRUN=1`: Prints every execution step without downloading or executing anything:
  ```powershell
  $env:DESKFLOW_DRYRUN = "1"
  powershell -File install.ps1
  ```

---

## 5. Storage Locations

Deskflow stores all user data and configuration strictly inside standard user-scoped directories:
- **Windows:**
  - Data: `%LOCALAPPDATA%\deskflow\deskflow`
  - Config: `%LOCALAPPDATA%\deskflow\deskflow`
- **macOS:**
  - Data: `~/Library/Application Support/deskflow`
  - Config: `~/Library/Application Support/deskflow`

---

## 6. Troubleshooting

- **Antivirus or Firewall Flagging:** Some corporate endpoint protection systems restrict `irm ... | iex` execution. The safe fallback is to download `install.ps1`, open and inspect the contents in an editor, and execute `powershell -File install.ps1`.
- **Terminal Display:** On Windows, use Windows Terminal for full UTF-8 box-drawing and color support. If running in legacy `conhost`, pass `--plain` to force ASCII layout.
- **TLS Errors:** Ensure TLS 1.2+ is enabled on Windows PowerShell (`[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12`).

---

## 7. Uninstallation

Because Deskflow requires no installer registration or registry keys, removal is clean and instant:

```bash
# If installed via DESKFLOW_INSTALL=1:
uv tool uninstall deskflow

# Clean uv runtime cache:
uv cache clean

# Remove local configuration and telemetry logs:
# Windows (PowerShell):
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\deskflow"
# macOS:
rm -rf ~/Library/Application\ Support/deskflow
```

