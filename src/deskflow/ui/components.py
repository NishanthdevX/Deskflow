"""Terminal Design v4 component primitives: responsive banner, status strip, KPI cards, Gauge, BarCompare, ProcessTree, Verdict, KeyHints."""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from deskflow.ui.theme import (
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_GREY,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    DOMAIN_BORDER_STYLES,
    STATUS_STYLES,
)


def banner(version: str, width: int = 100, height: int = 34, plain: bool = False) -> RenderableType:
    """Render the official block-letter DESKFLOW banner for the home screen.

    Tiers:
    - height >= 34 and width >= 80: full 6-row block banner.
    - height >= 26 and width >= 60: spaced wordmark.
    - else: plain compact title.
    """
    if plain or width < 80 or height < 26:
        text = Text()
        text.append(f"DESKFLOW v{version}\n", style="bold")
        text.append("Device health scan with ML slowdown prediction\n", style="dim")
        return text

    if height < 34:
        # Spaced wordmark
        text = Text()
        text.append("D E S K F L O W", style=f"bold {COLOR_CYAN}")
        text.append(f"  v{version}\n", style=COLOR_GREY)
        text.append("Device health scan with ML slowdown prediction\n", style="dim")
        return text

    # Official 6-row block-letter DESKFLOW in cyan (D16 fix)
    art = (
        "  ██████╗  ███████╗ ███████╗ ██╗  ██╗ ███████╗ ██╗       ██████╗  ██╗    ██╗\n"
        "  ██╔══██╗ ██╔════╝ ██╔════╝ ██║ ██╔╝ ██╔════╝ ██║      ██╔═══██╗ ██║    ██║\n"
        "  ██║  ██║ █████╗   ███████╗ █████═╝  █████╗   ██║      ██║   ██║ ██║ █╗ ██║\n"
        "  ██║  ██║ ██╔══╝   ╚════██║ ██╔═██╗  ██╔══╝   ██║      ██║   ██║ ██║███╗██║\n"
        "  ██████╔╝ ███████╗ ███████║ ██║  ██╗ ██║      ███████╗ ╚██████╔╝ ╚███╔███╔╝\n"
        "  ╚═════╝  ╚══════╝ ╚══════╝ ╚═╝  ╚═╝ ╚═╝      ╚══════╝  ╚═════╝   ╚══╝╚══╝ "
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
    """Render a clean single-row status strip with fixed padding and no garbled titles."""
    table = Table.grid(expand=True, padding=(0, 2))
    col_count = 5 if duration_s is not None else 4
    for _ in range(col_count):
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
    """Render a bounded horizontal progress/usage bar."""
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

    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    elif len(values) < width:
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


class Gauge:
    """Gauge showing an anomaly score against a threshold with marker and status."""

    def __init__(
        self,
        value: float,
        threshold: float,
        max_value: float = 1.0,
        width: int = 14,
        plain: bool = False,
    ):
        self.value = value
        self.threshold = threshold
        self.max_value = max(0.001, max_value)
        self.width = width
        self.plain = plain

    @property
    def status(self) -> str:
        """Return normal, borderline (within 5% of threshold), or unusual."""
        if self.threshold <= 0:
            return "normal"
        diff_pct = (self.value - self.threshold) / self.threshold
        if abs(diff_pct) <= 0.05:
            return "borderline"
        if self.value > self.threshold:
            return "unusual"
        return "normal"

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        v_norm = max(0.0, min(1.0, self.value / self.max_value))
        t_norm = max(0.0, min(1.0, self.threshold / self.max_value))

        v_pos = int(round(v_norm * self.width))
        t_pos = int(round(t_norm * self.width))

        track = ["░"] * self.width if not self.plain else ["-"] * self.width
        for i in range(min(v_pos, self.width)):
            track[i] = "█" if not self.plain else "#"

        if 0 <= t_pos < self.width:
            track[t_pos] = "│" if not self.plain else "|"

        bar_str = "".join(track)
        status_word = self.status
        if self.plain:
            yield Text(f"{bar_str} {self.value:.2f} ({status_word})")
        else:
            style = COLOR_GREEN if status_word == "normal" else (COLOR_YELLOW if status_word == "borderline" else COLOR_RED)
            yield Text.from_markup(f"[{style}]{bar_str}[/] [{COLOR_GREY}]{self.value:.2f} ({status_word})[/]")


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
    """Render a structured 4-line responsive KPI card."""
    accent = DOMAIN_BORDER_STYLES.get(domain, COLOR_GREY)
    lines: list[str] = []

    # Line 1: Header
    if plain:
        lines.append(f"[{title}]")
    else:
        lines.append(f"[{accent}][{title}][/]")

    # Line 2: Hero value
    if plain:
        lines.append(f"{hero_value}  {context}".strip())
    else:
        lines.append(f"[{COLOR_WHITE}]{hero_value}[/]  [{COLOR_GREY}]{context}[/]".strip())

    # Line 3: Sub-metrics
    sub_str = " | ".join(sub_metrics) if sub_metrics else ""
    if plain:
        lines.append(sub_str)
    else:
        lines.append(f"[{COLOR_GREY}]{sub_str}[/]")

    # Line 4: Bounded bar
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


def generate_verdict_sentence(
    score: int,
    predicted_ratio: float,
    cause_tag: str,
    cause_conf: float,
    first_action_title: str | None = None,
    potential_mb: float = 0.0,
) -> str:
    """Generate a clean rule-based 3-second verdict sentence."""
    pot_str = f"{potential_mb:.0f} MB of cleanable files found" if potential_mb > 0 else "no clutter found"
    if score >= 80:
        return (
            f"Your laptop is running normally. Predicted slowdown {predicted_ratio:.2f}x. "
            f"Nothing urgent; {pot_str}."
        )
    action_part = f"Suggested action: {first_action_title}." if first_action_title else "Run Fix Center to optimize."
    cause_str = cause_tag.replace("_", " ")
    return (
        f"Performance pressure detected. Predicted slowdown {predicted_ratio:.2f}x. "
        f"Likely cause: {cause_str} ({cause_conf:.0%}). {action_part}"
    )


def render_key_hints(hints: list[tuple[str, str]], plain: bool = False) -> Table:
    """Render a fixed key navigation bar."""
    grid = Table.grid(expand=True, padding=(0, 2))
    for _ in hints:
        grid.add_column()

    cells: list[str] = []
    for key, desc in hints:
        if plain:
            cells.append(f"[{key}] {desc}")
        else:
            cells.append(f"[{COLOR_CYAN}][{key}][/{COLOR_CYAN}] [{COLOR_GREY}]{desc}[/]")
    grid.add_row(*cells)
    return grid


def render_process_tree(
    heavy_tasks: list[Any],
    max_items: int = 5,
    plain: bool = False,
) -> Tree:
    """Render heavy tasks grouped by app as a Rich ProcessTree, marking locked system apps."""
    tree = Tree("[bold cyan]Active Background Tasks[/bold cyan]" if not plain else "Active Background Tasks")
    if not heavy_tasks:
        tree.add("[dim]No heavy background tasks detected[/dim]" if not plain else "No heavy background tasks detected")
        return tree

    for t in heavy_tasks[:max_items]:
        name = getattr(t, "name", str(t))
        cpu = getattr(t, "cpu_percent", 0.0)
        rss = getattr(t, "rss_bytes", 0) / (1024 * 1024)
        pcount = getattr(t, "process_count", 1)

        label = f"{name} ({pcount} proc) - {cpu:.1f}% CPU, {rss:.0f} MB RAM"
        node = tree.add(label)
        pids = getattr(t, "pids", ())
        if pids:
            pid_str = ", ".join(f"PID {p}" for p in pids[:3])
            node.add(f"[dim]{pid_str}[/dim]" if not plain else pid_str)

    return tree


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
