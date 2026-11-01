"""Platform-abstraction subpackage.

All OS-specific code (ctypes, winreg, macOS IOKit calls) lives here behind
a common interface. Nothing outside this subpackage may import ctypes or
winreg, or check sys.platform. Implementation is added in Phase 1.
"""
