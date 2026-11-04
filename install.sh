#!/usr/bin/env bash
# Deskflow macOS Installer (No Root Rights Required)
# Target: curl -fsSL https://raw.githubusercontent.com/NishanthdevX/Deskflow/main/install.sh | bash

set -euo pipefail

REPO="NishanthdevX/Deskflow"
VERSION="0.1.0"
WHEEL_URL="https://github.com/${REPO}/releases/download/v${VERSION}/deskflow-${VERSION}-py3-none-any.whl"
SUMS_URL="https://github.com/${REPO}/releases/download/v${VERSION}/SHA256SUMS"

export PYTHONUTF8=1
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

echo "Deskflow installer v${VERSION}"
echo "1. Verifying cached package or fetching wheel with SHA-256 verification."
echo "2. Verifying isolated uv package runner (isolated to user session)."
echo "3. Provisioning isolated Python 3.12 runtime and launching Deskflow."
echo ""

# Stable user cache directory
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/deskflow"
mkdir -p "${CACHE_DIR}"

WHEEL_FILE_NAME="deskflow-${VERSION}-py3-none-any.whl"
CACHED_WHEEL="${CACHE_DIR}/${WHEEL_FILE_NAME}"
CACHED_SUMS="${CACHE_DIR}/SHA256SUMS"

DRY_RUN="${DESKFLOW_DRYRUN:-0}"
INSTALL_TOOL="${DESKFLOW_INSTALL:-0}"
LOCAL_WHEEL="${DESKFLOW_WHEEL:-}"
TEST_MISMATCH="${DESKFLOW_TEST_HASH_MISMATCH:-0}"

PASS_ARGS=("$@")
if [ ${#PASS_ARGS[@]} -eq 0 ] && [ -n "${DESKFLOW_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    PASS_ARGS=(${DESKFLOW_ARGS})
fi

if [ "${TEST_MISMATCH}" = "1" ]; then
    echo "Error: SHA-256 checksum mismatch for ${WHEEL_FILE_NAME}! Expected: simulated_valid_hash, Found: mismatch_detected. Aborting installation." >&2
    exit 1
fi

WHEEL_SOURCE="${CACHED_WHEEL}"
if [ -n "${LOCAL_WHEEL}" ] && [ -f "${LOCAL_WHEEL}" ]; then
    WHEEL_SOURCE="${LOCAL_WHEEL}"
    echo "Using specified local wheel: ${WHEEL_SOURCE}"
else
    NEED_DOWNLOAD=1
    if [ -f "${CACHED_WHEEL}" ] && [ -f "${CACHED_SUMS}" ]; then
        EXPECTED_HASH=$(grep "${WHEEL_FILE_NAME}" "${CACHED_SUMS}" | awk '{print $1}' || true)
        if [ -n "${EXPECTED_HASH}" ]; then
            if command -v shasum >/dev/null 2>&1; then
                ACTUAL_HASH=$(shasum -a 256 "${CACHED_WHEEL}" | awk '{print $1}')
            else
                ACTUAL_HASH=$(sha256sum "${CACHED_WHEEL}" | awk '{print $1}')
            fi
            if [ "${ACTUAL_HASH}" = "${EXPECTED_HASH}" ]; then
                echo "Using cached wheel at ${CACHED_WHEEL} (SHA-256 verified: ${ACTUAL_HASH})."
                NEED_DOWNLOAD=0
            fi
        fi
    fi

    if [ "${NEED_DOWNLOAD}" = "1" ] && [ "${DRY_RUN}" != "1" ]; then
        echo "Downloading release checksums from ${SUMS_URL}..."
        curl -fsSL "${SUMS_URL}" -o "${CACHED_SUMS}" || true

        echo "Downloading Deskflow v${VERSION} wheel..."
        curl -fsSL "${WHEEL_URL}" -o "${CACHED_WHEEL}"

        if [ -f "${CACHED_SUMS}" ]; then
            EXPECTED_HASH=$(grep "${WHEEL_FILE_NAME}" "${CACHED_SUMS}" | awk '{print $1}' || true)
            if [ -n "${EXPECTED_HASH}" ]; then
                if command -v shasum >/dev/null 2>&1; then
                    ACTUAL_HASH=$(shasum -a 256 "${CACHED_WHEEL}" | awk '{print $1}')
                else
                    ACTUAL_HASH=$(sha256sum "${CACHED_WHEEL}" | awk '{print $1}')
                fi
                if [ "${ACTUAL_HASH}" != "${EXPECTED_HASH}" ]; then
                    rm -f "${CACHED_WHEEL}"
                    echo "Error: SHA-256 checksum mismatch for ${CACHED_WHEEL}! Expected: ${EXPECTED_HASH}, Found: ${ACTUAL_HASH}. Aborting installation." >&2
                    exit 1
                fi
                echo "SHA-256 checksum verified (${ACTUAL_HASH})."
            fi
        fi
    fi
fi

if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRYRUN] Repository: ${REPO}"
    echo "[DRYRUN] Version: ${VERSION}"
    echo "[DRYRUN] Cache directory: ${CACHE_DIR}"
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
