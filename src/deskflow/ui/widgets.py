"""Small reusable Rich widgets used by command-line dashboards."""

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def score_bar(score: int, width: int = 18, plain: bool = False) -> str:
    """Render a bounded score bar using Unicode or ASCII characters."""
    filled = max(0, min(width, round((score / 100) * width)))
    if plain:
        return "#" * filled + "-" * (width - filled)
    return "█" * filled + "░" * (width - filled)


def comparison_bar(value: float, maximum: float, width: int = 12, plain: bool = False) -> str:
    """Render a normalized comparison bar."""
    filled = 0 if maximum <= 0 else max(0, min(width, round(value / maximum * width)))
    if plain:
        return "#" * filled + "-" * (width - filled)
    return "▰" * filled + "▱" * (width - filled)


def titled_panel(title: str, body: str | Text | Table, plain: bool = False) -> Panel:
    """Return a bordered titled panel with conservative status-safe styling."""
    return Panel(body, title=title, box=box.ASCII if plain else box.ROUNDED, expand=True)


def key_value_table(rows: list[tuple[str, str]], plain: bool = False) -> Table:
    """Build a compact two-column key/value table."""
    table = Table(box=box.ASCII if plain else box.SIMPLE, show_header=False, pad_edge=False)
    table.add_column("Key", style="dim")
    table.add_column("Value")
    for key, value in rows:
        table.add_row(key, value)
    return table
