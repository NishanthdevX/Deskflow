"""Console layout breakpoints, height tiers, and responsive grid helpers."""

from __future__ import annotations

import shutil

from rich.console import Console, RenderableType
from rich.table import Table


class LayoutBreakpoints:
    """Width and height tiers for responsive terminal rendering."""

    WIDE = 120
    NORMAL = 100
    COMPACT = 80

    HEIGHT_FULL_BANNER = 34
    HEIGHT_WORDMARK = 26


def get_terminal_dimensions(console: Console | None = None) -> tuple[int, int]:
    """Return effective terminal (width, height), falling back safely to (100, 30)."""
    if console is not None and console.width:
        cols = console.width
        rows = console.height or 30
        return cols, rows
    cols, rows = shutil.get_terminal_size(fallback=(100, 30))
    return cols, rows


def is_wide(width: int) -> bool:
    """Return True if console width >= 120."""
    return width >= LayoutBreakpoints.WIDE


def is_normal(width: int) -> bool:
    """Return True if console width >= 100."""
    return width >= LayoutBreakpoints.NORMAL


def is_compact(width: int) -> bool:
    """Return True if console width is between 80 and 99."""
    return LayoutBreakpoints.COMPACT <= width < LayoutBreakpoints.NORMAL


def is_stacked(width: int) -> bool:
    """Return True if console width < 80."""
    return width < LayoutBreakpoints.COMPACT


def make_card_grid(cards: list[RenderableType], width: int = 100) -> Table:
    """Arrange cards responsively:

    - Width >= 100: single horizontal row (1xN).
    - Width 80..99: 2x2 grid (if 4 cards) or 2 columns.
    - Width < 80: stacked vertically (1 column).
    """
    grid = Table.grid(expand=True, padding=(0, 1))

    if width >= LayoutBreakpoints.NORMAL:
        for _ in range(len(cards)):
            grid.add_column(ratio=1)
        grid.add_row(*cards)
    elif width >= LayoutBreakpoints.COMPACT and len(cards) >= 4:
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        # 2x2
        grid.add_row(cards[0], cards[1])
        grid.add_row(cards[2], cards[3])
        if len(cards) > 4:
            for extra in cards[4:]:
                grid.add_row(extra)
    else:
        grid.add_column(ratio=1)
        for c in cards:
            grid.add_row(c)

    return grid

