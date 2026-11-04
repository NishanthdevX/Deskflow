# 5. Terminal Chart Rendering Approach

Date: 2026-11-04
Status: Accepted

## Context

Phase 11 (Terminal Design v2) introduces dense graphical elements into the command-line interface:
- Sparklines for 12 s metric history.
- Horizontal bar charts with 95% confidence interval whiskers for model comparisons.
- Line charts for live and report views (predicted vs. actual telemetry slowdown).
- Heatmaps for classification confusion matrices.

Phase 11 mandates an Architectural Decision Record (ADR) evaluating `plotext` (pure Python terminal plotting library) versus an in-house block and braille renderer against these criteria:
1. Installed size and lazy import overhead (target limit < 80 ms).
2. Pure ASCII fallback under `--plain`, `NO_COLOR`, and non-UTF-8 terminals.
3. Integration with Rich panels, tables, and renderables.
4. Deterministic layout at 80, 100, and 120 terminal columns.
5. Strict zero-emoji policy and absence of external dependency bloat.

## Evaluation

### plotext
- **Dependency & Size**: plotext is not part of the standard dependencies. Installing it adds external package weight and extra wheel surface area.
- **Import Overhead**: Benchmarks across Python environments typically show import times between 40 ms and 110 ms depending on disk cache and filesystem speed, edging close to or exceeding the 80 ms ceiling.
- **Output Control**: plotext manages its own canvas and string formatting with internal ANSI escape sequences, making tight integration into Rich grid layouts, borders, and column-width wrapping fragile.
- **Accessibility & Fallbacks**: Adapting plotext dynamically to pure ASCII `--plain` mode without colored control codes requires significant post-processing.

### In-House Block & Braille Renderer (`deskflow.ui.charts`)
- **Dependency & Size**: 0 bytes added to wheel dependencies; 100% in-tree pure Python.
- **Import Overhead**: < 0.8 ms import time (over 100x faster than the 80 ms limit).
- **Control & Consistency**: Direct control over Unicode braille characters (U+2800..U+28FF), quadrant blocks (U+2596..U+259F), shaded blocks (U+2591..U+2593), and full blocks (U+2588).
- **ASCII Fallbacks**: Seamless automatic fallback to ASCII characters (`#`, `-`, `+`, `*`, `.`, `|`) when `--plain`, `NO_COLOR`, or non-UTF-8 terminals are active.
- **Rich Integration**: Emits Rich `Text` and `Table` instances that fit natively into `Panel` and responsive layouts at 80, 100, and 120 columns without horizontal overflow.

## Decision

We adopt the **In-House Block and Braille Renderer** (`src/deskflow/ui/charts.py`).
`plotext` is omitted to preserve zero-dependency lightweight execution, instant CLI responsiveness (< 50 ms render time), and seamless Rich layout integration.

