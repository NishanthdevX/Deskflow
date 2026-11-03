"""Rendering tests for shared dashboard widgets and the home launcher."""

from rich.console import Console

from deskflow.cli import show_home_menu
from deskflow.ui.widgets import scan_dashboard


def _render(plain: bool, width: int) -> str:
    """Render a representative scan dashboard at one console width."""
    console = Console(record=True, width=width, no_color=plain)
    panels = scan_dashboard(
        "machine-id",
        "battery",
        "XGBoost",
        True,
        85,
        "Good",
        1.2,
        "healthy",
        0.8,
        False,
        ["No significant factors"],
        [("Memory", "Good", "8 GB free", 50.0)],
        [("Clean temporary files", "120 MB measured", None)],
        14.0,
        plain,
    )
    for panel in panels:
        console.print(panel)
    return console.export_text()


def test_dashboard_renders_at_supported_widths() -> None:
    """Dashboard panels render at the documented terminal widths."""
    for width in (80, 100, 120):
        output = _render(False, width)
        assert "SCAN SUMMARY" in output
        assert "DEMO" in output


def test_plain_dashboard_is_ascii() -> None:
    """Plain mode is restricted to printable ASCII output."""
    for width in (80, 100, 120):
        assert _render(True, width).isascii()


def test_home_menu_quit_choice(monkeypatch) -> None:
    """The home menu accepts the non-destructive quit selection."""
    responses = iter(["5"])
    show_home_menu(lambda _prompt: next(responses))
