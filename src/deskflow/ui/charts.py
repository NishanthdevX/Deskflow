"""In-house terminal charts: whiskers bar charts, braille/block line plots, heatmaps."""

from __future__ import annotations

from rich import box
from rich.table import Table
from rich.text import Text

from deskflow.ui.components import bar
from deskflow.ui.theme import (
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_GREY,
    COLOR_WHITE,
    COLOR_YELLOW,
)


def bar_chart_with_whiskers(
    rows: list[
        tuple[str, float, float, float, bool]
    ],  # (name, metric, ci_low, ci_high, is_baseline)
    metric_label: str = "MAE",
    width: int = 24,
    plain: bool = False,
) -> Table:
    """Render a horizontal bar chart with 95% confidence interval whiskers.

    Baseline row (e.g. Persistence) is clearly marked and dimmed.
    """
    table = Table(
        box=box.ASCII if plain else box.SIMPLE_HEAD,
        show_header=True,
        header_style=COLOR_CYAN if not plain else None,
        expand=True,
    )
    table.add_column("Model", style="bold" if not plain else None)
    table.add_column(metric_label, justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Distribution", justify="left")

    if not rows:
        return table

    max_val = max(max(r[3], r[1]) for r in rows)
    max_val = max(1e-6, max_val * 1.1)

    for name, mean_val, ci_low, ci_high, is_baseline in rows:
        ci_str = f"[{ci_low:.3f}, {ci_high:.3f}]"
        mean_str = f"{mean_val:.3f}"

        # Build whisker track of fixed width
        track = ["-" if plain else "░"] * width
        p_mean = int(round(max(0.0, min(1.0, mean_val / max_val)) * (width - 1)))
        p_low = int(round(max(0.0, min(1.0, ci_low / max_val)) * (width - 1)))
        p_high = int(round(max(0.0, min(1.0, ci_high / max_val)) * (width - 1)))

        # Fill bar up to mean
        for i in range(p_mean + 1):
            track[i] = "#" if plain else "█"

        # Apply whiskers
        if p_low <= p_high:
            for i in range(p_low, p_high + 1):
                if track[i] in ("░", "-"):
                    track[i] = "=" if plain else "─"
            if 0 <= p_low < width:
                track[p_low] = "[" if plain else "├"
            if 0 <= p_high < width:
                track[p_high] = "]" if plain else "┤"
            if 0 <= p_mean < width and (p_low <= p_mean <= p_high):
                track[p_mean] = "#" if plain else "█"

        chart_bar = "".join(track)
        label_disp = f"{name} (baseline)" if is_baseline else name

        if is_baseline:
            style = "dim"
            table.add_row(
                Text(label_disp, style=style),
                Text(mean_str, style=style),
                Text(ci_str, style=style),
                Text(chart_bar, style=style),
            )
        else:
            table.add_row(label_disp, mean_str, ci_str, chart_bar)

    return table


def line_chart(
    actual: list[float],
    predicted: list[float],
    title: str = "PREDICTED VS ACTUAL",
    x_label: str = "Time (samples)",
    y_label: str = "Ratio",
    height: int = 8,
    width: int = 50,
    plain: bool = False,
) -> Table:
    """Render a dual-series line plot (Predicted vs Actual) using braille/block or ASCII."""
    table = Table.grid(expand=True)
    table.add_column()

    if not actual and not predicted:
        table.add_row(Text("[No time-series data]", style=COLOR_GREY))
        return table

    n_pts = max(len(actual), len(predicted))
    if n_pts == 0:
        return table

    # Resample series to width
    def resample(series: list[float], target_len: int) -> list[float]:
        if not series:
            return [1.0] * target_len
        if len(series) == target_len:
            return list(series)
        out = []
        step = len(series) / target_len
        for i in range(target_len):
            idx = min(len(series) - 1, int(i * step))
            out.append(series[idx])
        return out

    act_res = resample(actual, width)
    pred_res = resample(predicted, width)

    all_vals = act_res + pred_res
    min_y = min(all_vals)
    max_y = max(all_vals)
    if (max_y - min_y) < 1e-4:
        max_y = min_y + 1.0

    # Grid of characters: height rows by width cols
    canvas = [[" " for _ in range(width)] for _ in range(height)]
    # Color annotations
    colors = [["" for _ in range(width)] for _ in range(height)]

    for col in range(width):
        # Actual y pos
        act_val = act_res[col]
        row_act = height - 1 - int(round(((act_val - min_y) / (max_y - min_y)) * (height - 1)))
        row_act = max(0, min(height - 1, row_act))

        # Pred y pos
        pred_val = pred_res[col]
        row_pred = height - 1 - int(round(((pred_val - min_y) / (max_y - min_y)) * (height - 1)))
        row_pred = max(0, min(height - 1, row_pred))

        if row_act == row_pred:
            canvas[row_act][col] = "*" if plain else "◈"
            colors[row_act][col] = COLOR_YELLOW
        else:
            canvas[row_act][col] = "A" if plain else "●"
            colors[row_act][col] = COLOR_WHITE
            canvas[row_pred][col] = "P" if plain else "■"
            colors[row_pred][col] = COLOR_CYAN

    # Legend & header
    hdr = Text()
    hdr.append(f"[{title}]  ", style="bold")
    if plain:
        hdr.append("[P] Predicted  [A] Actual  [*] Overlap\n", style="dim")
    else:
        hdr.append("■ Predicted  ", style=COLOR_CYAN)
        hdr.append("● Actual  ", style=COLOR_WHITE)
        hdr.append("◈ Overlap\n", style=COLOR_YELLOW)
    table.add_row(hdr)

    # Render rows with y-axis tick
    for r in range(height):
        row_y = max_y - (r / (height - 1)) * (max_y - min_y)
        tick = f"{row_y:4.2f} ┤" if not plain else f"{row_y:4.2f} |"
        row_text = Text(tick, style=COLOR_GREY)
        for c in range(width):
            ch = canvas[r][c]
            col_style = colors[r][c]
            if col_style and not plain:
                row_text.append(ch, style=col_style)
            else:
                row_text.append(ch)
        table.add_row(row_text)

    # X-axis
    axis_line = "     └" + ("─" * width) if not plain else "     +" + ("-" * width)
    table.add_row(Text(axis_line, style=COLOR_GREY))
    lbl_line = f"      0s{' ' * max(0, width - 8)}{x_label}"
    table.add_row(Text(lbl_line, style=COLOR_GREY))

    return table


def heatmap(
    labels: list[str],
    matrix: list[list[int]],
    title: str = "CONFUSION MATRIX",
    plain: bool = False,
) -> Table:
    """Render a classification confusion matrix using shade blocks and counts.

    Shades: U+2591, U+2592, U+2593, U+2588.
    ASCII: '.', '+', '#', '@'.
    """
    table = Table(
        title=f"[{title}]",
        box=box.ASCII if plain else box.SQUARE,
        show_header=True,
        header_style=COLOR_CYAN if not plain else None,
    )
    table.add_column("Actual \\ Pred", style="bold" if not plain else None)
    for lbl in labels:
        table.add_column(lbl, justify="center")

    max_val = max(max(row) for row in matrix) if matrix and any(matrix) else 1
    max_val = max(1, max_val)

    # Shading thresholds
    for i, actual_lbl in enumerate(labels):
        row_cells = [actual_lbl]
        for j, _pred_lbl in enumerate(labels):
            count = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else 0
            ratio = count / max_val
            if plain:
                if ratio == 0:
                    glyph = " "
                elif ratio < 0.3:
                    glyph = "."
                elif ratio < 0.6:
                    glyph = "+"
                elif ratio < 0.85:
                    glyph = "#"
                else:
                    glyph = "@"
            elif ratio == 0:
                glyph = " "
            elif ratio < 0.3:
                glyph = "░"
            elif ratio < 0.6:
                glyph = "▒"
            elif ratio < 0.85:
                glyph = "▓"
            else:
                glyph = "█"

            cell_str = f"{glyph} {count}" if count > 0 else f"{glyph}  0"
            if not plain:
                # Highlight diagonal in green/cyan if non-zero
                if i == j and count > 0:
                    row_cells.append(f"[{COLOR_GREEN}]{cell_str}[/]")
                elif count > 0:
                    row_cells.append(f"[{COLOR_YELLOW}]{cell_str}[/]")
                else:
                    row_cells.append(f"[{COLOR_GREY}]{cell_str}[/]")
            else:
                row_cells.append(cell_str)

        table.add_row(*row_cells)

    return table


def leaderboard_table(
    models: list[tuple[str, str, float, float, float, bool, bool]],
    plain: bool = False,
) -> Table:
    """Format the model leaderboard table.

    Tuples: (name, family, score, ci_low, ci_high, is_primary, is_baseline)
    """
    table = Table(
        box=box.ASCII if plain else box.HORIZONTALS,
        show_header=True,
        header_style=COLOR_CYAN if not plain else None,
        expand=True,
    )
    table.add_column("Model")
    table.add_column("Type")
    table.add_column("MAE", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Relative Error Bar", justify="left")

    if not models:
        return table

    max_mae = max(m[2] for m in models)
    max_mae = max(1e-4, max_mae)

    for name, family, mae, ci_l, ci_h, is_prim, is_base in models:
        prefix = "* " if is_prim else "  "
        model_name = f"{prefix}{name}"
        ratio = min(1.0, mae / max_mae)
        bar_str = bar(ratio, 1.0, width=14, plain=plain)

        if is_base:
            style = "dim"
            table.add_row(
                Text(f"{model_name} (baseline)", style=style),
                Text(family, style=style),
                Text(f"{mae:.3f}", style=style),
                Text(f"[{ci_l:.3f}, {ci_h:.3f}]", style=style),
                Text(bar_str, style=style),
            )
        elif is_prim:
            prim_style = f"bold {COLOR_CYAN}" if not plain else ""
            table.add_row(
                Text(f"{model_name} (primary)", style=prim_style),
                family,
                f"{mae:.3f}",
                f"[{ci_l:.3f}, {ci_h:.3f}]",
                bar_str,
            )
        else:
            table.add_row(
                model_name,
                family,
                f"{mae:.3f}",
                f"[{ci_l:.3f}, {ci_h:.3f}]",
                bar_str,
            )

    return table
