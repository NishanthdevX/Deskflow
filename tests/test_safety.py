"""Unit tests for safety rules and process protection checks."""

from deskflow.safety import is_protected


def test_protected_system_pids() -> None:
    """Fixed system PIDs 0, 1, and 4 must always be protected."""
    assert is_protected("idle", 0, "root")
    assert is_protected("init", 1, "root")
    assert is_protected("System", 4, "SYSTEM")


def test_protected_current_pid_and_parent() -> None:
    """The current running process and any ancestor process must be protected."""
    current_pid = 1234
    parent_pids = {500, 100}

    # Current process
    assert is_protected("deskflow.exe", 1234, "alice", current_pid=current_pid)
    # Ancestor process
    assert is_protected("pwsh.exe", 500, "alice", current_pid=current_pid, parent_pids=parent_pids)
    assert is_protected(
        "terminal.exe", 100, "alice", current_pid=current_pid, parent_pids=parent_pids
    )


def test_protected_process_names_case_insensitive() -> None:
    """Protected names must match case-insensitively."""
    protected = {"explorer.exe", "launchd", "dwm.exe"}

    assert is_protected(
        "EXPLORER.EXE",
        9999,
        "alice",
        current_pid=1,
        current_user="alice",
        parent_pids=set(),
        protected_names=protected,
    )
    assert is_protected(
        "LaunchD",
        9999,
        "alice",
        current_pid=1,
        current_user="alice",
        parent_pids=set(),
        protected_names=protected,
    )


def test_process_owned_by_other_user() -> None:
    """Processes not owned by current user must be protected."""
    assert is_protected(
        "worker.exe",
        8888,
        "bob",
        current_pid=1,
        current_user="alice",
        parent_pids=set(),
        protected_names=set(),
    )


def test_process_with_unknown_user_is_protected() -> None:
    """Processes with None username (e.g. AccessDenied) must be protected by default."""
    assert is_protected(
        "unknown.exe",
        7777,
        None,
        current_pid=1,
        current_user="alice",
        parent_pids=set(),
        protected_names=set(),
    )


def test_unprotected_normal_user_process() -> None:
    """A regular user process not in protected names or parent chain is unprotected."""
    assert not is_protected(
        "slack.exe",
        6543,
        "alice",
        current_pid=1,
        current_user="alice",
        parent_pids={100},
        protected_names={"explorer.exe"},
    )
