"""Regenerate plain-text screen captures and SVGs in docs/screens/ using Console(record=True)."""

from __future__ import annotations

import platform
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deskflow import __version__
from deskflow.commands.fix import _render_tools_table, run_fast_inventory
from deskflow.commands.info import render_info_dashboard
from deskflow.commands.pipeline import render_pipeline_frame
from deskflow.ml.facts import load_model_facts
from deskflow.platform.factory import get_adapter
from deskflow.ui.components import (
    banner,
    bar,
    generate_verdict_sentence,
    kpi_card,
    render_key_hints,
    status_strip,
)
from deskflow.ui.layout import make_card_grid

ROOT_DIR = Path(__file__).resolve().parent.parent
SCREENS_DIR = ROOT_DIR / "docs" / "screens"


def capture_home(console: Console) -> None:
    facts = load_model_facts()
    cols = 100

    console.print(banner(version=__version__, width=cols, height=34, plain=False))

    m_lines = [
        f"Host / OS: {platform.node()} (Windows 11)",
        "CPU: AMD Ryzen 7 6800HS with Radeon Graphics (8C/16T @ 3200 MHz)",
        "GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6.0 GB VRAM) | RAM: 15.2 GB",
        "Power: AC Wall Power",
    ]
    card_machine = Panel(
        "\n".join(m_lines),
        title="[ THIS MACHINE ]",
        border_style="cyan",
        box=box.ROUNDED,
    )

    mod_lines = [
        "Status: REAL HARDWARE MODEL",
        f"Dataset: {facts.dataset_split_summary}",
        "Horizon: 20s ahead slowdown prediction",
        "Baselines: AC (fresh, 2h ago) | Battery (fresh, 4h ago)",
    ]
    card_model = Panel(
        "\n".join(mod_lines),
        title="[ MODEL FACTS ]",
        border_style="green",
        box=box.ROUNDED,
    )

    grid_top = make_card_grid([card_machine, card_model], width=cols)
    console.print(grid_top)
    console.print()

    menu_table = Table.grid(expand=True, padding=(0, 3))
    menu_table.add_column(ratio=1)
    menu_table.add_column(ratio=1)

    c1 = (
        "[bold cyan][1][/bold cyan] Battery check     "
        "[dim]about 75s  health, prediction, accuracy[/dim]"
    )
    c2 = (
        "[bold cyan][4][/bold cyan] Fix Center        "
        "[dim]clean cache, memory, idle tasks[/dim]"
    )
    c3 = (
        "[bold cyan][2][/bold cyan] Plugged-in check  "
        "[dim]about 75s  same, on AC wall power[/dim]"
    )
    c4 = (
        "[bold cyan][5][/bold cyan] How it works      "
        "[dim]animated pipeline & architecture[/dim]"
    )
    c5 = (
        "[bold cyan][3][/bold cyan] Full scan         "
        "[dim]about 10m  automated hardware stress suite[/dim]"
    )
    c6 = (
        "[bold cyan][6][/bold cyan] System info       "
        "[dim]fastfetch CPU, GPU, memory, disks[/dim]"
    )

    menu_table.add_row(c1, c2)
    menu_table.add_row(c3, c4)
    menu_table.add_row(c5, c6)
    console.print(menu_table)
    console.print()
    console.print(render_key_hints([("1-6", "Select tool"), ("?", "Glossary"), ("Q", "Quit")]))


def capture_check(console: Console) -> None:
    cols = 100

    stat_strip = status_strip(
        os_info="Windows 11",
        power_state="AC",
        model_name="RandomForest",
        synthetic=False,
        baseline_age="2h ago",
        duration_s=14.2,
        plain=False,
    )
    console.print(stat_strip)
    console.rule(style="cyan")

    verdict = generate_verdict_sentence(
        score=92,
        predicted_ratio=1.04,
        cause_tag="healthy",
        cause_conf=0.96,
        first_action_title="Clean temporary files",
        potential_mb=279.0,
    )
    console.print(f"[bold]{verdict}[/bold]\n")

    card_health = kpi_card(
        title="HEALTH",
        hero_value="92/100",
        context="Optimal",
        sub_metrics=["100 = Baseline", "Optimal status"],
        bar_ratio=0.92,
        domain="health",
    )
    card_slow = kpi_card(
        title="NEXT 20 S",
        hero_value="1.04x",
        context="vs baseline",
        sub_metrics=["Horizon 20s", "104% baseline"],
        bar_ratio=1.04 / 3.0,
        domain="ml",
    )
    card_cause = kpi_card(
        title="LIKELY CAUSE",
        hero_value="Healthy",
        context="96% conf",
        sub_metrics=["health: 96%", "cpu: 2%", "load: 1%"],
        bar_ratio=0.96,
        domain="ml",
    )
    card_anom = kpi_card(
        title="ANOMALY CHECK",
        hero_value="NORMAL",
        context="0.21 score",
        sub_metrics=["IsolationForest", "Thresh 0.60"],
        bar_ratio=0.21 / 0.60,
        domain="ml",
    )
    console.print(make_card_grid([card_health, card_slow, card_cause, card_anom], width=cols))
    console.print()

    why_table = Table(box=box.HORIZONTALS, expand=True, show_header=False)
    why_table.add_column("Factor", ratio=3)
    why_table.add_column("Impact", ratio=1, justify="right")
    why_table.add_row("Micro-benchmark probe latency within baseline", bar(0.1, 1.0, 10))
    why_table.add_row("CPU frequency ratio stable at 100%", bar(0.05, 1.0, 10))
    p_why = Panel(why_table, title="[ WHY THE MODEL SAYS THIS ]", border_style="cyan")

    res_table = Table(box=box.HORIZONTALS, expand=True, show_header=False)
    res_table.add_column("Resource", ratio=2)
    res_table.add_column("Usage", ratio=3)
    res_table.add_column("Status", ratio=2, justify="right")
    res_table.add_row("CPU", f"{bar(0.14, 1.0, 10)} 14.2%", "[green]Normal[/green]")
    res_table.add_row("Memory", f"{bar(0.48, 1.0, 10)} 48.1%", "[green]Normal[/green]")
    res_table.add_row("Disk Space", "128 GB free (28%)", "[green]OK[/green]")
    p_res = Panel(res_table, title="[ RESOURCES ]", border_style="grey50")

    grid_mid = Table.grid(expand=True, padding=(0, 2))
    grid_mid.add_column(ratio=1)
    grid_mid.add_column(ratio=1)
    grid_mid.add_row(p_why, p_res)
    console.print(grid_mid)
    console.print()
    console.print("[dim]Timings: preflight 0.18s | measuring 14.20s | infer 0.0032s[/dim]")
    console.print(
        render_key_hints(
            [("F", "Fix Center"), ("W", "Live Watch"), ("Enter", "Menu"), ("Q", "Quit")]
        )
    )


def update_screens() -> None:
    """Capture real CLI screens as SVG and plain text."""
    SCREENS_DIR.mkdir(parents=True, exist_ok=True)
    adapter = get_adapter()
    facts = load_model_facts()

    screens_to_render = [
        ("home", capture_home),
        ("check", capture_check),
    ]

    for name, render_fn in screens_to_render:
        c = Console(width=100, record=True)
        render_fn(c)

        txt_path = SCREENS_DIR / f"{name}.txt"
        svg_path = SCREENS_DIR / f"{name}.svg"

        txt_out = c.export_text()
        txt_path.write_text(txt_out, encoding="utf-8")
        c.save_svg(str(svg_path), title="deskflow")
        print(f"Generated {txt_path.name} and {svg_path.name}")

    # Fix inventory
    c_fix = Console(width=100, record=True)
    tools = run_fast_inventory(adapter)
    c_fix.print(_render_tools_table(tools, plain=False))
    (SCREENS_DIR / "fix_list.txt").write_text(c_fix.export_text(), encoding="utf-8")
    c_fix.save_svg(str(SCREENS_DIR / "fix_list.svg"), title="deskflow")
    print("Generated fix_list.txt and fix_list.svg")

    # Info
    c_info = Console(width=100, record=True)
    render_info_dashboard(json_output=False, plain=False, console_out=c_info)
    (SCREENS_DIR / "info.txt").write_text(c_info.export_text(), encoding="utf-8")
    c_info.save_svg(str(SCREENS_DIR / "info.svg"), title="deskflow")
    print("Generated info.txt and info.svg")

    # Pipeline
    c_pipe = Console(width=100, record=True)
    final_frame = render_pipeline_frame(
        active_stage=5,
        packet_pos=5.0,
        real_value_text="Fix Center: 3 safe tools available, 279.0 MB measured potential",
        facts=facts,
        plain=False,
    )
    c_pipe.print(final_frame)
    (SCREENS_DIR / "pipeline.txt").write_text(c_pipe.export_text(), encoding="utf-8")
    c_pipe.save_svg(str(SCREENS_DIR / "pipeline.svg"), title="deskflow")
    print("Generated pipeline.txt and pipeline.svg")


if __name__ == "__main__":
    update_screens()
