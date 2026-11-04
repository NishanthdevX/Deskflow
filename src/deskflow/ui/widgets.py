"""Small reusable Rich widgets used by command-line dashboards and scan views."""

from __future__ import annotations

import shutil
from typing import Any

from rich import box
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table

from deskflow.ui.components import (
    badge,
    bar,
    domain_panel,
    kpi_card,
    sparkline,
    status_strip,
)
from deskflow.ui.theme import (
    COLOR_CYAN,
    COLOR_GREY,
    DOMAIN_BORDER_STYLES,
    STATUS_STYLES,
)


def score_bar(score: int, width: int = 18, plain: bool = False) -> str:
    """Render a bounded score bar using Unicode or ASCII characters."""
    return bar(score / 100.0, 1.0, width=width, plain=plain)


def comparison_bar(value: float, maximum: float, width: int = 12, plain: bool = False) -> str:
    """Render a normalized comparison bar."""
    return bar(value, maximum, width=width, plain=plain)


def titled_panel(
    title: str, body: RenderableType, plain: bool = False, domain: str = "system"
) -> Panel:
    """Return a bordered titled panel with conservative status-safe styling."""
    return domain_panel(title, body, domain=domain, plain=plain)


def key_value_table(rows: list[tuple[str, str]], plain: bool = False) -> Table:
    """Build a compact two-column key/value table."""
    table = Table(box=box.ASCII if plain else box.SIMPLE, show_header=False, pad_edge=False)
    table.add_column("Key", style=COLOR_GREY if not plain else None)
    table.add_column("Value")
    for key, value in rows:
        table.add_row(key, value)
    return table


def status_style(status: str) -> str:
    """Map documented Good/Fair/Poor status labels to semantic styles."""
    return STATUS_STYLES.get(status, COLOR_GREY)


def scan_dashboard(
    baseline_age: str,
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
    anomaly_score: float = 0.0,
    anomaly_threshold: float = 0.5,
    heavy_tasks: list[Any] | None = None,
    fix_tools: list[Any] | None = None,
    resources: list[tuple[str, float, list[float]]] | None = None,
    timings: dict[str, float] | None = None,
    os_info: str = "Windows",
) -> list[Panel]:
    """Build the structured scan view strictly matching terminal design v2."""
    cols, _ = shutil.get_terminal_size(fallback=(100, 30))

    # 1. Status strip panel [SCAN SUMMARY: DESKFLOW SCAN]
    status_tbl = status_strip(
        os_info=os_info,
        power_state=power_state,
        model_name=model_name,
        synthetic=synthetic,
        baseline_age=baseline_age,
        duration_s=elapsed_s,
        plain=plain,
    )
    status_panel = domain_panel(
        "[SCAN SUMMARY: DESKFLOW SCAN]",
        status_tbl,
        domain="system",
        plain=plain,
    )

    # 2. KPI row / cards [HEALTH AND PREDICTION]
    # Card 1: HEALTH
    health_card = kpi_card(
        title="HEALTH",
        hero_value=f"{score}/100",
        context=health_label,
        sub_metrics=["100 = Baseline", f"{health_label} performance"],
        bar_ratio=score / 100.0,
        domain="health",
        plain=plain,
    )

    # Card 2: NEXT 20 S
    slow_pct = int(round(predicted_ratio * 100))
    slow_card = kpi_card(
        title="NEXT 20 S",
        hero_value=f"{predicted_ratio:.2f}x",
        context="vs baseline",
        sub_metrics=["Horizon 20s", f"{slow_pct}% baseline ratio"],
        bar_ratio=min(1.0, predicted_ratio / 3.0),
        domain="ml",
        plain=plain,
    )

    # Card 3: LIKELY CAUSE
    issue_card = kpi_card(
        title="LIKELY CAUSE",
        hero_value=issue_tag,
        context=f"{confidence:.0%} conf",
        sub_metrics=["Telemetry tag", f"{confidence:.0%} confidence"],
        bar_ratio=confidence,
        domain="ml",
        plain=plain,
    )

    # Card 4: UNUSUAL
    anom_max = max(0.01, anomaly_threshold)
    anom_card = kpi_card(
        title="UNUSUAL",
        hero_value="Yes" if anomaly else "No",
        context=f"{anomaly_score:.2f} score",
        sub_metrics=["IsolationForest", f"Thresh {anomaly_threshold:.2f}"],
        bar_ratio=min(1.0, anomaly_score / anom_max),
        domain="ml",
        plain=plain,
    )

    if cols >= 100:
        kpi_grid = Table.grid(expand=True)
        for _ in range(4):
            kpi_grid.add_column(ratio=1)
        kpi_grid.add_row(health_card, slow_card, issue_card, anom_card)
        kpi_renderable: RenderableType = kpi_grid
    else:
        # Stacked for narrow widths (< 100 cols)
        kpi_stack = Table.grid(expand=True)
        kpi_stack.add_column(ratio=1)
        kpi_stack.add_row(health_card)
        kpi_stack.add_row(slow_card)
        kpi_stack.add_row(issue_card)
        kpi_stack.add_row(anom_card)
        kpi_renderable = kpi_stack

    kpi_panel = domain_panel(
        "[HEALTH AND PREDICTION]",
        kpi_renderable,
        domain="health" if score >= 80 else ("actions" if score >= 60 else "health"),
        plain=plain,
    )

    # 3. Split: [ WHY ] (Top factors) and [ RESOURCES ]
    why_table = Table(
        box=box.ASCII if plain else box.HORIZONTALS,
        show_header=False,
        expand=True,
    )
    why_table.add_column("Factor", ratio=3, overflow="fold")
    why_table.add_column("Impact", ratio=1, justify="right", overflow="crop")
    for i, reason in enumerate(reasons):
        frac = max(0.25, 1.0 - (i * 0.25))
        why_table.add_row(reason, comparison_bar(frac, 1.0, 10, plain))
    if not reasons:
        why_table.add_row("All metrics within baseline range", comparison_bar(0.0, 1.0, 10, plain))

    # Resources: CPU, memory, disk, swap bars + sparklines
    res_table = Table(
        box=box.ASCII if plain else box.HORIZONTALS,
        show_header=False,
        expand=True,
    )
    res_table.add_column("Resource", ratio=2, overflow="fold")
    res_table.add_column("Usage", ratio=3, overflow="crop")
    res_table.add_column("Trend (12s)", ratio=2, justify="right", overflow="crop")

    if resources:
        for res_name, res_ratio, spark_vals in resources:
            pct = int(round(max(0.0, min(1.0, res_ratio)) * 100))
            bar_text = f"{bar(res_ratio, 1.0, width=12, plain=plain)} {pct:>3}%"
            spark_text = sparkline(spark_vals, width=10, plain=plain)
            res_table.add_row(res_name, bar_text, spark_text)
    else:
        # Fallback default values
        res_table.add_row(
            "CPU",
            f"{bar(0.15, 1.0, 12, plain)}  15%",
            sparkline([10, 12, 15, 14, 15, 16], 10, plain),
        )
        res_table.add_row(
            "Memory",
            f"{bar(0.50, 1.0, 12, plain)}  50%",
            sparkline([48, 49, 50, 50, 50, 50], 10, plain),
        )
        res_table.add_row(
            "Disk Busy",
            f"{bar(0.05, 1.0, 12, plain)}   5%",
            sparkline([2, 4, 3, 5, 2, 5], 10, plain),
        )
        res_table.add_row(
            "Swap", f"{bar(0.01, 1.0, 12, plain)}   1%", sparkline([1, 1, 1, 1, 1, 1], 10, plain)
        )

    why_panel = domain_panel("[ WHY: TOP FACTORS ]", why_table, domain="ml", plain=plain)
    res_panel = domain_panel("[ RESOURCES ]", res_table, domain="system", plain=plain)

    if cols >= 100:
        split_grid = Table.grid(expand=True)
        split_grid.add_column(ratio=1)
        split_grid.add_column(ratio=1)
        split_grid.add_row(why_panel, res_panel)
        split_renderable: RenderableType = split_grid
    else:
        split_stack = Table.grid(expand=True)
        split_stack.add_column(ratio=1)
        split_stack.add_row(why_panel)
        split_stack.add_row(res_panel)
        split_renderable = split_stack

    # 4. [ CATEGORIES ] table
    cat_table = Table(box=box.ASCII if plain else box.SQUARE, expand=True)
    cat_table.add_column("Category", overflow="fold")
    cat_table.add_column("Status", overflow="fold")
    cat_table.add_column("Details", overflow="crop")
    for name, state, detail, value in categories:
        details = detail + (" " + comparison_bar(value, 100, 8, plain) if value > 0 else "")
        cat_table.add_row(name, f"[{status_style(state)}]{state}[/]", details)
    cat_panel = domain_panel("[CATEGORIES]", cat_table, domain="health", plain=plain)

    # 5. [ HEAVY TASKS ] panel (top 3 by CPU and top 3 by memory)
    ht_table = Table(
        box=box.ASCII if plain else box.HORIZONTALS,
        show_header=True,
        header_style=COLOR_CYAN if not plain else None,
        expand=True,
    )
    ht_table.add_column("Task", overflow="fold")
    ht_table.add_column("Metric", justify="right", overflow="crop")
    ht_table.add_column("Detail", justify="right", overflow="crop")

    if heavy_tasks:
        for t in heavy_tasks[:6]:
            t_name = getattr(t, "name", str(t))
            cpu = getattr(t, "cpu_percent", 0.0)
            rss = getattr(t, "rss_bytes", 0) / (1024 * 1024)
            pcount = getattr(t, "process_count", 1)
            ht_table.add_row(
                f"{t_name} ({pcount} proc)",
                f"{cpu:.1f}% CPU",
                f"{rss:.1f} MB RAM",
            )
    else:
        ht_table.add_row("No heavy background tasks detected", "-", "-")

    heavy_panel = domain_panel("[ HEAVY TASKS ]", ht_table, domain="system", plain=plain)

    # 6. [ ACTIONS ] panel
    # Numbered fix tools with risk badge and measured potential; always shown,
    # with "No action recommended" in header when healthy.
    is_healthy = score >= 80 and not recommendations
    actions_title = "[ ACTIONS: No action recommended ]" if is_healthy else "[ ACTIONS ]"

    actions_table = Table(
        box=box.ASCII if plain else box.HORIZONTALS,
        show_header=True,
        header_style=COLOR_CYAN if not plain else None,
        expand=True,
    )
    actions_table.add_column("#", justify="right", width=3, overflow="crop")
    actions_table.add_column("Tool", ratio=2, overflow="fold")
    actions_table.add_column("Risk", justify="center", width=10, overflow="crop")
    actions_table.add_column("Measured Potential", ratio=2, overflow="crop")
    actions_table.add_column("Scope / Action", ratio=3, overflow="fold")

    if fix_tools:
        for tool in fix_tools:
            risk_badge = badge(tool.risk, tool.risk, plain=plain)
            actions_table.add_row(
                str(tool.number),
                tool.title,
                risk_badge,
                tool.potential_str,
                tool.what_it_changes,
            )
    elif recommendations:
        for i, (title, evidence, warning) in enumerate(recommendations, 1):
            warn_str = f" (Warning: {warning})" if warning else ""
            actions_table.add_row(
                str(i), title, badge("RECOMMENDED", "Warn", plain=plain), evidence, warn_str
            )
    else:
        # Standard default 6 tools inventory representation
        actions_table.add_row(
            "1",
            "Clean temporary files",
            badge("SAFE", "Safe", plain=plain),
            "0.0 MB cleanable",
            "Deletes files > 24h in temp roots",
        )
        actions_table.add_row(
            "2",
            "Clean app caches",
            badge("SAFE", "Safe", plain=plain),
            "0.0 MB cleanable",
            "Browser caches (pages load slower once)",
        )
        actions_table.add_row(
            "3",
            "Empty Recycle Bin",
            badge("CAUTION", "Caution", plain=plain),
            "0.0 MB (0 items)",
            "Empties Recycle Bin permanently",
        )
        actions_table.add_row(
            "4",
            "Trim memory",
            badge("SAFE", "Safe", plain=plain),
            "Reclaim working sets",
            "Reclaims memory (temporary effect)",
        )
        actions_table.add_row(
            "5",
            "Close idle apps",
            badge("CAUTION", "Caution", plain=plain),
            "0 idle app groups",
            "Closes idle background windowed apps",
        )
        actions_table.add_row(
            "6",
            "End heavy tasks",
            badge("CAUTION", "Caution", plain=plain),
            "0 heavy tasks",
            "Ends top non-protected heavy tasks",
        )

    footer_note = None
    if timings:
        footer_note = (
            f"Timings: measure {timings.get('measuring_s', 0):.2f}s | "
            f"storage {timings.get('storage_s', 0):.2f}s | "
            f"infer {timings.get('inference_s', 0):.4f}s | "
            f"total {timings.get('total_s', 0):.2f}s"
        )

    actions_panel = domain_panel(
        actions_title,
        actions_table,
        domain="actions" if not is_healthy else "health",
        plain=plain,
        subtitle=footer_note,
    )

    return [
        status_panel,
        kpi_panel,
        Panel(
            split_renderable,
            box=box.ASCII if plain else box.ROUNDED,
            border_style=COLOR_GREY if plain else DOMAIN_BORDER_STYLES["system"],
        ),
        cat_panel,
        heavy_panel,
        actions_panel,
    ]
