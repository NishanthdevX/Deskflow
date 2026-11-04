#!/usr/bin/env bash
# Deskflow macOS Installer (No Root Rights Required)
# Target: curl -fsSL https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.sh | bash

set -euo pipefail

REPO="NishanthdevX/Deskflow"
VERSION="0.1.0"
WHEEL_URL="https://github.com/${REPO}/releases/download/v${VERSION}/deskflow-${VERSION}-py3-none-any.whl"

export PYTHONUTF8=1
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

echo "Deskflow installer v${VERSION}"
echo "1. Verifying or fetching uv package runner (isolated to user session)."
echo "2. Provisioning isolated Python 3.12 runtime and dependencies on first launch."
echo "3. Launching Deskflow. No administrator rights or permanent system changes are required."
echo ""

DRY_RUN="${DESKFLOW_DRYRUN:-0}"
INSTALL_TOOL="${DESKFLOW_INSTALL:-0}"
LOCAL_WHEEL="${DESKFLOW_WHEEL:-}"

WHEEL_SOURCE="${WHEEL_URL}"
if [ -n "${LOCAL_WHEEL}" ] && [ -f "${LOCAL_WHEEL}" ]; then
    WHEEL_SOURCE="${LOCAL_WHEEL}"
fi

PASS_ARGS=("$@")
if [ ${#PASS_ARGS[@]} -eq 0 ] && [ -n "${DESKFLOW_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    PASS_ARGS=(${DESKFLOW_ARGS})
fi

if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRYRUN] Repository: ${REPO}"
    echo "[DRYRUN] Version: ${VERSION}"
    echo "[DRYRUN] Wheel source: ${WHEEL_SOURCE}"
    echo "[DRYRUN] Tool install mode: ${INSTALL_TOOL}"
    echo "[DRYRUN] Arguments to pass: ${PASS_ARGS[*]:-none}"
    echo "[DRYRUN] Launch command: uvx --python 3.12 --from ${WHEEL_SOURCE} deskflow ${PASS_ARGS[*]:-}"
    exit 0
fi

# Check for uv
if ! command -v uv >/dev/null 2>&1; then
    if [ -x "${HOME}/.local/bin/uv" ]; then
        export PATH="${HOME}/.local/bin:${PATH}"
    elif [ -x "${HOME}/.cargo/bin/uv" ]; then
        export PATH="${HOME}/.cargo/bin:${PATH}"
    fi
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Fetching isolated uv package manager..."
    export UV_NO_MODIFY_PATH=1
    if ! curl -fsSL https://astral.sh/uv/install.sh | bash; then
        echo "Error: Failed to download uv. Please verify internet connectivity or proxy configuration." >&2
        exit 1
    fi
    if [ -x "${HOME}/.local/bin/uv" ]; then
        export PATH="${HOME}/.local/bin:${PATH}"
    elif [ -x "${HOME}/.cargo/bin/uv" ]; then
        export PATH="${HOME}/.cargo/bin:${PATH}"
    else
        echo "Error: uv was installed but the binary could not be found in user PATH." >&2
        exit 1
    fi
fi

# Run or install
if [ "${INSTALL_TOOL}" = "1" ]; then
    echo "Installing Deskflow to user tool environment..."
    uv tool install --python 3.12 --from "${WHEEL_SOURCE}" deskflow --force
    echo "Deskflow installed. Run 'deskflow' from your terminal."
else
    if [ ${#PASS_ARGS[@]} -gt 0 ]; then
        exec uvx --python 3.12 --from "${WHEEL_SOURCE}" deskflow "${PASS_ARGS[@]}"
    else
        exec uvx --python 3.12 --from "${WHEEL_SOURCE}" deskflow
    fi
fi

