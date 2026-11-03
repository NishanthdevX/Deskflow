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


def status_style(status: str) -> str:
    """Map documented Good/Fair/Poor status labels to semantic styles."""
    return {"Good": "spring_green2", "Fair": "yellow1", "Poor": "red1"}.get(status, "grey50")


def scan_dashboard(
    machine_id: str,
    power_state: str,
    model_name: str,
    synthetic: bool,
    score: int,
    health_label: str,
    predicted_ratio: float,
    issue_tag: str,
    confidence: float,
    anomaly: bool,
    reasons: list[str],
    categories: list[tuple[str, str, str, float]],
    recommendations: list[tuple[str, str, str | None]],
    elapsed_s: float,
    plain: bool = False,
) -> list[Panel]:
    """Build the structured scan view described in the terminal design document."""
    status = Table.grid(expand=True)
    for _ in range(5):
        status.add_column()
    status.add_row(
        f"STATUS: {health_label}",
        f"POWER: {power_state}",
        f"MODEL: {model_name}{' DEMO' if synthetic else ''}",
        f"BASELINE: {machine_id[:8]}",
        f"SCAN: {elapsed_s:.1f}s",
    )
    kpis = Table.grid(expand=True)
    for _ in range(4):
        kpis.add_column(ratio=1)
    health = f"[HEALTH]\n{score}/100 {health_label}\n{score_bar(score, 14, plain)} {score}%"
    slowdown = f"[SLOWDOWN]\n{predicted_ratio:.2f}x in 20 s\nbaseline ratio\n"
    slowdown += comparison_bar(predicted_ratio, 3, 14, plain)
    issue = f"[ISSUE]\n{issue_tag}\nconfidence {confidence:.0%}\n"
    issue += comparison_bar(confidence, 1, 14, plain)
    anomaly_text = f"[ANOMALY]\n{'Detected' if anomaly else 'None'}\nmodel threshold\n"
    anomaly_text += comparison_bar(1 if anomaly else 0, 1, 14, plain)
    kpis.add_row(health, slowdown, issue, anomaly_text)
    factors = Table(box=box.ASCII if plain else box.HORIZONTALS, show_header=False)
    factors.add_column("Factor")
    for reason in reasons:
        factors.add_row(reason)
    category_table = Table(box=box.ASCII if plain else box.SQUARE)
    category_table.add_column("Category")
    category_table.add_column("Status")
    category_table.add_column("Details")
    for name, state, detail, value in categories:
        details = detail + " " + comparison_bar(value, 100, 8, plain)
        category_table.add_row(name, f"[{status_style(state)}]{state}[/]", details)
    fixes = Table(box=box.ASCII if plain else box.HORIZONTALS, show_header=False)
    fixes.add_column("Fix")
    for title, evidence, warning in recommendations:
        fixes.add_row(f"{title}: {evidence}" + (f" Warning: {warning}" if warning else ""))
    return [
        titled_panel("[SCAN SUMMARY: DESKFLOW SCAN]", status, plain),
        titled_panel("[HEALTH AND PREDICTION]", kpis, plain),
        titled_panel("[TOP FACTORS]", factors, plain),
        titled_panel("[CATEGORIES]", category_table, plain),
        titled_panel("[RECOMMENDED FIXES]", fixes, plain),
    ]
