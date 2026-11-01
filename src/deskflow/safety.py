"""Safety rules and process protection checks."""

import functools
import getpass
import os
from collections.abc import Set

from deskflow.platform import get_adapter

_HARDCODED_PROTECTED_PIDS = frozenset({0, 1, 4})


@functools.lru_cache(maxsize=32)
def _get_parent_pids(pid: int) -> frozenset[int]:
    """Inspect and return the set of ancestor PIDs for a given PID."""
    parents: set[int] = set()
    try:
        import psutil

        proc = psutil.Process(pid)
        for parent in proc.parents():
            parents.add(parent.pid)
    except Exception:
        pass
    return frozenset(parents)


@functools.lru_cache(maxsize=1)
def _get_current_username() -> str:
    """Return current username, lowercase."""
    try:
        return getpass.getuser().lower()
    except Exception:
        return ""


def is_protected(
    name: str,
    pid: int,
    username: str | None = None,
    current_pid: int | None = None,
    current_user: str | None = None,
    parent_pids: Set[int] | None = None,
    protected_names: Set[str] | None = None,
) -> bool:
    """Determine whether a process is protected and must never be terminated.

    Returns True for:
    - Fixed system PIDs: 0, 1, 4
    - The current running DESKFLOW process
    - Any process in the parent chain of the current process (terminals, shells)
    - Any process name on the platform's protected names list (case-insensitive)
    - Any process not owned by the current user (or where ownership cannot be verified)

    Args:
        name: The process executable name (e.g. 'explorer.exe' or 'launchd').
        pid: The process identifier.
        username: The user account owning the process, or None if unknown.
        current_pid: Override for the current PID (defaults to os.getpid()).
        current_user: Override for the current username (defaults to getpass.getuser()).
        parent_pids: Override for the set of parent PIDs.
        protected_names: Override for protected process names set.

    Returns:
        True if the process is protected; False otherwise.
    """
    # 1. Fixed system PIDs
    if pid in _HARDCODED_PROTECTED_PIDS:
        return True

    # 2. Current running process
    eff_current_pid = os.getpid() if current_pid is None else current_pid
    if pid == eff_current_pid:
        return True

    # 3. Parent chain
    eff_parent_pids: Set[int] = (
        _get_parent_pids(eff_current_pid) if parent_pids is None else parent_pids
    )
    if pid in eff_parent_pids:
        return True

    # 4. Protected process names list
    eff_protected_names: Set[str]
    if protected_names is None:
        try:
            eff_protected_names = get_adapter().protected_process_names()
        except Exception:
            eff_protected_names = frozenset()
    else:
        eff_protected_names = protected_names

    if name.lower() in {p.lower() for p in eff_protected_names}:
        return True

    # 5. User ownership check
    eff_current_user = _get_current_username() if current_user is None else current_user.lower()
    if not username:
        # Cannot determine ownership (e.g. AccessDenied on Windows/macOS); protect it by default
        return True

    clean_user = username.lower()
    # On Windows, usernames may be in DOMAIN\User format
    if "\\" in clean_user:
        clean_user = clean_user.split("\\")[-1]

    return bool(eff_current_user and clean_user != eff_current_user)
