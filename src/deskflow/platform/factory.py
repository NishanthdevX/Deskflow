"""Platform adapter factory."""

import sys

from deskflow.platform.base import PlatformAdapter
from deskflow.platform.macos import MacOSAdapter
from deskflow.platform.windows import WindowsAdapter


def get_adapter() -> PlatformAdapter:
    """Return the PlatformAdapter implementation appropriate for the current OS.

    Raises:
        NotImplementedError: If running on an unsupported operating system.
    """
    if sys.platform.startswith("win"):
        return WindowsAdapter()
    if sys.platform == "darwin":
        return MacOSAdapter()
    raise NotImplementedError(
        f"Operating system '{sys.platform}' is not supported. "
        "Deskflow currently supports Windows 10/11 and macOS 12+."
    )
