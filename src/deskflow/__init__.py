"""deskflow package.

Exposes the package version read from installed package metadata so that
the CLI and any other consumer have a single authoritative source.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("deskflow")
except PackageNotFoundError:
    # Package is not installed (e.g. running from source without uv sync).
    __version__ = "0.0.0.unknown"
