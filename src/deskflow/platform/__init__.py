"""Platform abstraction layer.

Exports the get_adapter() factory and all public types. Nothing outside this
subpackage may import ctypes, winreg, or check sys.platform.
"""

from deskflow.platform.base import BatteryInfo, CacheLocation, CpuFreq, PlatformAdapter
from deskflow.platform.factory import get_adapter

__all__ = [
    "BatteryInfo",
    "CacheLocation",
    "CpuFreq",
    "PlatformAdapter",
    "get_adapter",
]
