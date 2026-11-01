"""Tests that every subpackage can be imported."""


def test_import_deskflow() -> None:
    """The top-level package must be importable."""
    import deskflow  # noqa: F401


def test_import_collector() -> None:
    import deskflow.collector  # noqa: F401


def test_import_features() -> None:
    import deskflow.features  # noqa: F401


def test_import_ml() -> None:
    import deskflow.ml  # noqa: F401


def test_import_health() -> None:
    import deskflow.health  # noqa: F401


def test_import_remediation() -> None:
    import deskflow.remediation  # noqa: F401


def test_import_evaluation() -> None:
    import deskflow.evaluation  # noqa: F401


def test_import_platform() -> None:
    import deskflow.platform  # noqa: F401


def test_import_cli() -> None:
    import deskflow.cli  # noqa: F401


def test_version_is_string() -> None:
    """__version__ must be a non-empty string."""
    from deskflow import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0
