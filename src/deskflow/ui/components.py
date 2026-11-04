"""Terminal Design v2 component primitives: banner, status strip, KPI cards, bars, badges."""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from deskflow.ui.theme import (
    COLOR_CYAN,
    COLOR_GREY,
    COLOR_WHITE,
    DOMAIN_BORDER_STYLES,
    STATUS_STYLES,
)


def banner(version: str, width: int = 100, plain: bool = False) -> RenderableType:
    """Render the official block-letter DESKFLOW banner for the home screen.

    Falls back to a plain single-line title when width < 80, plain is requested,
    or non-UTF-8 output is active.
    """
    if plain or width < 80:
        text = Text()
        text.append(f"DESKFLOW v{version}\n", style="bold")
        text.append("Device health scan with ML slowdown prediction\n", style="dim")
        return text

    # Dense block-letter DESKFLOW in cyan (fits within 80 cols)
    art = (
        "  ██████╗ ███████╗███████╗██╗  ██╗███████╗██╗      ██████╗ ██╗    ██╗\n"
        "  ██╔══██╗██╔════╝██╔════╝██║ ██╔╝██╔════╝██║     ██╔═══██╗██║    ██║\n"
        "  ██║  ██║█████╗  ███████╗█████═╝ █████╗  ██║     ██║   ██║██║ █╗ ██║\n"
        "  ██║  ██║██╔══╝  ╚════██║██╔═██╗ ██╔══╝  ██║     ██║   ██║██║███╗██║\n"
        "  ██████╔╝███████╗███████║██║  ██╗██║     ███████╗╚██████╔╝╚███╔███╔╝"
    )
    t = Text(art, style=COLOR_CYAN)
    sub = Text(
        f"\n  DESKFLOW v{version} - Device health scan with ML slowdown prediction\n",
        style=COLOR_GREY,
    )
    t.append_text(sub)
    return t


def status_strip(
    os_info: str,
    power_state: str,
    model_name: str,
    synthetic: bool,
    baseline_age: str,
    duration_s: float | None = None,
    plain: bool = False,
) -> Table:
    """Render a dense single-row status strip."""
    table = Table.grid(expand=True)
    for _ in range(5 if duration_s is not None else 4):
        table.add_column()

    os_cell = f"OS: {os_info}"
    pwr_cell = f"POWER: {power_state}"
    model_flag = " DEMO" if synthetic else ""
    model_cell = (
        f"MODEL: [{COLOR_CYAN}]{model_name}{model_flag}[/]"
        if not plain
        else f"MODEL: {model_name}{model_flag}"
    )
    base_cell = f"BASELINE: {baseline_age}"

    if duration_s is not None:
        dur_cell = f"SCAN: {duration_s:.1f}s"
        table.add_row(os_cell, pwr_cell, model_cell, base_cell, dur_cell)
    else:
        table.add_row(os_cell, pwr_cell, model_cell, base_cell)

    return table


def bar(value: float, maximum: float = 1.0, width: int = 14, plain: bool = False) -> str:
    """Render a bounded horizontal progress/usage bar.

    Unicode: filled U+2588, track U+2591.
    Plain ASCII: filled '#', track '-'.
    """
    if maximum <= 0:
        fraction = 0.0
    else:
        fraction = max(0.0, min(1.0, value / maximum))

    filled = round(fraction * width)
    unfilled = width - filled

    if plain:
        return "#" * filled + "-" * unfilled
    return "█" * filled + "░" * unfilled


def sparkline(values: list[float], width: int = 12, plain: bool = False) -> str:
    """Render a sparkline trend using Unicode block levels or ASCII steps."""
    if not values:
        return "-" * width if plain else "░" * width

    # Resample or truncate to requested width
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    elif len(values) < width:
        # Pad left with the first value or minimum
        min_v = min(values)
        sampled = [min_v] * (width - len(values)) + list(values)
    else:
        sampled = list(values)

    min_val = min(sampled)
    max_val = max(sampled)
    val_range = max_val - min_val

    if plain:
        chars = '_.-~*^"'
        out: list[str] = []
        for v in sampled:
            if val_range <= 1e-6:
                idx = 0
            else:
                ratio = (v - min_val) / val_range
                idx = min(len(chars) - 1, int(ratio * (len(chars) - 1)))
            out.append(chars[idx])
        return "".join(out)

    # Unicode sparkline blocks: U+2581 to U+2588 (8 heights)
    blocks = " ▂▃▄▅▆▇█"
    out_uni: list[str] = []
    for v in sampled:
        if val_range <= 1e-6:
            idx = 0
        else:
            ratio = (v - min_val) / val_range
            idx = min(len(blocks) - 1, int(ratio * (len(blocks) - 1)))
        out_uni.append(blocks[idx])
    return "".join(out_uni)


def badge(label: str, rating: str, plain: bool = False) -> str:
    """Format a rating/risk badge with semantic styling."""
    style = STATUS_STYLES.get(rating, COLOR_GREY)
    if plain:
        return f"[{rating.upper()}]"
    return f"[{style}][{label}][/]"


def kpi_card(
    title: str,
    hero_value: str,
    context: str,
    sub_metrics: list[str],
    bar_ratio: float | None = None,
    domain: str = "health",
    plain: bool = False,
    bar_width: int = 14,
) -> Panel:
    """Render a structured 4-line KPI card.

    Line 1: Header in domain accent colour.
    Line 2: Hero value in bold white with muted context.
    Line 3: Pipe-separated sub-metrics.
    Line 4: 14-char bar (if real fraction exists) with percentage.
    """
    accent = DOMAIN_BORDER_STYLES.get(domain, COLOR_GREY)
    lines: list[str] = []

    # Line 1
    if plain:
        lines.append(f"[{title}]")
    else:
        lines.append(f"[{accent}][{title}][/]")

    # Line 2
    if plain:
        lines.append(f"{hero_value}  {context}".strip())
    else:
        lines.append(f"[{COLOR_WHITE}]{hero_value}[/]  [{COLOR_GREY}]{context}[/]".strip())

    # Line 3
    sub_str = " | ".join(sub_metrics) if sub_metrics else ""
    if plain:
        lines.append(sub_str)
    else:
        lines.append(f"[{COLOR_GREY}]{sub_str}[/]")

    # Line 4
    if bar_ratio is not None:
        pct = int(round(max(0.0, min(1.0, bar_ratio)) * 100))
        rendered_bar = bar(bar_ratio, 1.0, width=bar_width, plain=plain)
        if plain:
            lines.append(f"{rendered_bar} {pct}%")
        else:
            lines.append(f"[{accent}]{rendered_bar}[/] [{COLOR_GREY}]{pct}%[/]")
    else:
        lines.append("")

    card_text = "\n".join(lines)
    return Panel(
        card_text,
        box=box.ASCII if plain else box.ROUNDED,
        border_style=COLOR_GREY if plain else accent,
        expand=True,
    )


def domain_panel(
    title: str,
    content: Any,
    domain: str = "system",
    plain: bool = False,
    subtitle: str | None = None,
) -> Panel:
    """Wrap content in a bordered panel with domain-specific border styling."""
    border_color = DOMAIN_BORDER_STYLES.get(domain, COLOR_GREY)
    return Panel(
        content,
        title=title,
        subtitle=subtitle,
        box=box.ASCII if plain else box.ROUNDED,
        border_style=COLOR_GREY if plain else border_color,
        expand=True,
    )


def key_value_block(
    rows: list[tuple[str, str, float | None]],
    plain: bool = False,
    bar_width: int = 12,
) -> Table:
    """Format key-value rows with grey labels, bold white values, and optional bars."""
    table = Table.grid(expand=True, padding=(0, 2))
    table.add_column("Key", style=COLOR_GREY if not plain else None)
    table.add_column("Value", style=COLOR_WHITE if not plain else None)
    table.add_column("Bar", justify="right")

    for key, value, ratio in rows:
        if ratio is not None:
            bar_str = bar(ratio, 1.0, width=bar_width, plain=plain)
            pct = int(round(max(0.0, min(1.0, ratio)) * 100))
            bar_cell = f"{bar_str} {pct}%"
        else:
            bar_cell = ""
        table.add_row(key, value, bar_cell)

    return table
