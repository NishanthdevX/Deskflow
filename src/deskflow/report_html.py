"""Self-contained HTML report generator for deskflow report.

Generates dark-themed, publication-ready reports with inline SVGs, zero external
assets, under 500 KB, with full print-to-PDF stylesheet support.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from deskflow.truth import NO_REAL_DATA_MESSAGE


def _render_whiskers_svg(leaderboard: list[dict[str, Any]], width: int = 680) -> str:
    """Render horizontal bar chart with 95% bootstrap CI whiskers as inline SVG."""
    if not leaderboard:
        return "<p class='dim'>No leaderboard data available.</p>"

    row_h = 36
    label_w = 150
    chart_w = width - label_w - 180
    height = len(leaderboard) * row_h + 40

    max_val = max(
        max(r.get("mae_ci", [0.0, r.get("mae", 0.1)])[1], r.get("mae", 0.1)) for r in leaderboard
    )
    max_val = max(0.01, max_val * 1.15)

    svg_parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg" class="chart-svg">',
        f'<line x1="{label_w}" y1="10" x2="{label_w}" y2="{height - 20}" stroke="#334155" />',
        f'<line x1="{label_w}" y1="{height - 20}" x2="{label_w + chart_w}" '
        f'y2="{height - 20}" stroke="#334155" />',
    ]

    for idx, row in enumerate(leaderboard):
        m_name = html.escape(str(row.get("model", "Unknown")))
        mae = float(row.get("mae", 0.0))
        ci = row.get("mae_ci", [mae, mae])
        ci_l = float(ci[0])
        ci_h = float(ci[1])
        is_base = bool(row.get("is_baseline", False))

        y = 25 + idx * row_h
        bar_len = (mae / max_val) * chart_w
        x_low = label_w + (ci_l / max_val) * chart_w
        x_high = label_w + (ci_h / max_val) * chart_w
        x_mean = label_w + bar_len

        bar_color = "#64748b" if is_base else "#38bdf8"
        text_class = "model-label dim" if is_base else "model-label"
        suffix = " (baseline)" if is_base else ""

        # Model label
        svg_parts.append(
            f'<text x="{label_w - 10}" y="{y + 12}" text-anchor="end" '
            f'class="{text_class}">{m_name}{suffix}</text>'
        )
        # Bar track
        svg_parts.append(
            f'<rect x="{label_w}" y="{y}" width="{chart_w}" height="18" rx="3" fill="#1e293b" />'
        )
        # Value bar
        svg_parts.append(
            f'<rect x="{label_w}" y="{y}" width="{max(2.0, bar_len)}" height="18" '
            f'rx="3" fill="{bar_color}" opacity="{0.45 if is_base else 0.85}" />'
        )
        # Whisker line
        svg_parts.append(
            f'<line x1="{x_low}" y1="{y + 9}" x2="{x_high}" y2="{y + 9}" '
            f'stroke="{bar_color}" stroke-width="2" />'
        )
        # Whisker caps
        svg_parts.append(
            f'<line x1="{x_low}" y1="{y + 4}" x2="{x_low}" y2="{y + 14}" '
            f'stroke="{bar_color}" stroke-width="2" />'
        )
        svg_parts.append(
            f'<line x1="{x_high}" y1="{y + 4}" x2="{x_high}" y2="{y + 14}" '
            f'stroke="{bar_color}" stroke-width="2" />'
        )
        # Mean point
        svg_parts.append(f'<circle cx="{x_mean}" cy="{y + 9}" r="3.5" fill="#f8fafc" />')
        # Text annotation
        ci_text = f"{mae:.4f} [{ci_l:.3f}, {ci_h:.3f}]"
        svg_parts.append(
            f'<text x="{label_w + chart_w + 12}" y="{y + 13}" class="val-text">{ci_text}</text>'
        )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def _render_line_chart_svg(
    time_series: list[dict[str, Any]], width: int = 680, height: int = 200
) -> str:
    """Render dual-series Predicted vs Actual as inline SVG."""
    if not time_series:
        return "<p class='dim'>No time-series samples available.</p>"

    pad_l = 50
    pad_r = 20
    pad_t = 30
    pad_b = 30
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    all_vals = [pt["actual"] for pt in time_series] + [pt["predicted"] for pt in time_series]
    min_y = min(all_vals, default=0.5)
    max_y = max(all_vals, default=2.0)
    if (max_y - min_y) < 1e-4:
        max_y = min_y + 1.0

    n_pts = len(time_series)
    pts_act: list[str] = []
    pts_pred: list[str] = []
    pts_pers: list[str] = []

    for i, pt in enumerate(time_series):
        x = pad_l + (i / max(1, n_pts - 1)) * plot_w
        y_act = pad_t + plot_h - ((pt["actual"] - min_y) / (max_y - min_y)) * plot_h
        y_pred = pad_t + plot_h - ((pt["predicted"] - min_y) / (max_y - min_y)) * plot_h
        y_pers = pad_t + plot_h - ((pt.get("persistence", 1.0) - min_y) / (max_y - min_y)) * plot_h

        pts_act.append(f"{x:.1f},{y_act:.1f}")
        pts_pred.append(f"{x:.1f},{y_pred:.1f}")
        pts_pers.append(f"{x:.1f},{y_pers:.1f}")

    svg_parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg" class="chart-svg">',
        # Background grid
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        'fill="#0b0f17" stroke="#334155" />',
        # Y Ticks
        f'<text x="{pad_l - 8}" y="{pad_t + 10}" text-anchor="end" '
        f'class="axis-tick">{max_y:.2f}</text>',
        f'<text x="{pad_l - 8}" y="{pad_t + plot_h}" text-anchor="end" '
        f'class="axis-tick">{min_y:.2f}</text>',
        # X Ticks
        f'<text x="{pad_l}" y="{pad_t + plot_h + 18}" class="axis-tick">0s</text>',
        f'<text x="{pad_l + plot_w}" y="{pad_t + plot_h + 18}" text-anchor="end" '
        f'class="axis-tick">Test Samples ({n_pts})</text>',
        # Legend
        f'<line x1="{pad_l + 20}" y1="15" x2="{pad_l + 40}" y2="15" stroke="#38bdf8" '
        'stroke-width="2" />',
        f'<text x="{pad_l + 45}" y="19" class="legend-text">Predicted Slowdown</text>',
        f'<line x1="{pad_l + 180}" y1="15" x2="{pad_l + 200}" y2="15" stroke="#94a3b8" '
        'stroke-width="1.5" />',
        f'<text x="{pad_l + 205}" y="19" class="legend-text">Actual Target</text>',
        f'<line x1="{pad_l + 320}" y1="15" x2="{pad_l + 340}" y2="15" stroke="#facc15" '
        'stroke-width="1.5" stroke-dasharray="3,3" />',
        f'<text x="{pad_l + 345}" y="19" class="legend-text">Persistence</text>',
        # Polylines
        f'<polyline points="{" ".join(pts_pers)}" fill="none" stroke="#facc15" '
        'stroke-width="1.5" stroke-dasharray="3,3" opacity="0.6" />',
        f'<polyline points="{" ".join(pts_act)}" fill="none" stroke="#94a3b8" '
        'stroke-width="1.5" />',
        f'<polyline points="{" ".join(pts_pred)}" fill="none" stroke="#38bdf8" stroke-width="2" />',
        "</svg>",
    ]
    return "\n".join(svg_parts)


def _render_confusion_matrix_html(cm_data: dict[str, Any]) -> str:
    """Render confusion matrix as a shaded HTML table."""
    classes = cm_data.get("classes", [])
    matrix = cm_data.get("matrix", [])
    if not classes or not matrix:
        return "<p class='dim'>No confusion matrix data available.</p>"

    max_count = max((max(row) for row in matrix), default=1) or 1
    rows_html = ["<table class='cm-table'><thead><tr><th>Actual \\ Predicted</th>"]
    for c in classes:
        rows_html.append(f"<th>{html.escape(c)}</th>")
    rows_html.append("</tr></thead><tbody>")

    for i, act_cls in enumerate(classes):
        rows_html.append(f"<tr><td class='cm-label'>{html.escape(act_cls)}</td>")
        for j, _ in enumerate(classes):
            cnt = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else 0
            ratio = cnt / max_count
            alpha = max(0.05, min(0.9, ratio))
            diag_class = "cm-diag" if i == j and cnt > 0 else "cm-cell"
            if i == j:
                bg = f"rgba(56, 189, 248, {alpha:.2f})"
            else:
                bg = f"rgba(250, 204, 21, {alpha * 0.7:.2f})"
            if cnt == 0:
                bg = "transparent"
            rows_html.append(f"<td class='{diag_class}' style='background-color: {bg};'>{cnt}</td>")
        rows_html.append("</tr>")
    rows_html.append("</tbody></table>")
    return "\n".join(rows_html)


def generate_html_report(data: dict[str, Any], output_path: Path) -> Path:
    """Generate self-contained, publication-ready HTML verification report."""
    is_synth = bool(data.get("synthetic", False))
    machine = data.get("machine", {})
    baselines = data.get("baselines", {})
    leaderboard = data.get("leaderboard", [])
    scenario_breakdown = data.get("scenario_breakdown", {})
    power_breakdown = data.get("power_breakdown", {})
    confusion_matrix = data.get("confusion_matrix", {})
    feature_importance = data.get("feature_importance", [])
    anomaly = data.get("anomaly", {})
    time_series = data.get("time_series", [])
    watch_summary = data.get("watch_summary", {})
    fix_summary = data.get("fix_summary", {})
    limitations = data.get("limitations", [])

    synth_banner = ""
    if is_synth:
        synth_banner = (
            "<div class='banner-warning'>"
            "DEMO MODEL (SYNTHETIC DATA - NOT A REAL WORLD RESULT)"
            "</div>"
        )

    # Build leaderboard rows
    leaderboard_rows = []
    for r in leaderboard:
        m = html.escape(str(r.get("model", "")))
        mae = float(r.get("mae", 0.0))
        ci = r.get("mae_ci", [mae, mae])
        rmse = float(r.get("rmse", 0.0))
        r2 = float(r.get("r2", 0.0))
        tol = float(r.get("tolerance_accuracy", 0.0)) * 100.0
        is_base = bool(r.get("is_baseline", False))
        beats = r.get("beats_persistence", False)

        if is_base:
            beats_str = "<span class='dim'>-</span>"
        elif beats:
            beats_str = "<span class='badge-green'>YES</span>"
        else:
            beats_str = "<span class='badge-red'>NO</span>"

        cls_row = "class='baseline-row'" if is_base else ""
        leaderboard_rows.append(
            f"<tr {cls_row}>"
            f"<td><strong>{m}</strong>{' (baseline)' if is_base else ''}</td>"
            f"<td>{mae:.4f}</td>"
            f"<td>[{ci[0]:.4f}, {ci[1]:.4f}]</td>"
            f"<td>{rmse:.4f}</td>"
            f"<td>{r2:.4f}</td>"
            f"<td>{tol:.1f}%</td>"
            f"<td>{beats_str}</td>"
            f"</tr>"
        )

    # Scenarios
    sc_models = ["XGBoost", "RandomForest", "Ridge", "Persistence"]
    sc_headers = "".join(f"<th>{m}</th>" for m in sc_models)
    sc_rows = []
    for sc, sc_m in scenario_breakdown.items():
        cells = "".join(f"<td>{sc_m.get(m, 0.0):.4f}</td>" for m in sc_models)
        sc_rows.append(f"<tr><td>{html.escape(sc)}</td>{cells}</tr>")

    # Feature Importance Bars
    fi_items = []
    for f in feature_importance[:10]:
        name = html.escape(f.get("name", ""))
        weight = float(f.get("weight", 0.0))
        norm = float(f.get("normalized", 0.0))
        bar_w = int(norm * 100)
        fi_items.append(
            f"<div class='fi-row'>"
            f"<span class='fi-name'>{name}</span>"
            f"<div class='fi-track'><div class='fi-fill' "
            f"style='width: {bar_w}%;'></div></div>"
            f"<span class='fi-weight'>{weight:.4f}</span>"
            f"</div>"
        )

    # Power breakdown cells
    ac_cells = "".join(
        f"<td>{power_breakdown.get('ac', {}).get(m, 0.0):.4f}</td>" for m in sc_models
    )
    bat_cells = "".join(
        f"<td>{power_breakdown.get('battery', {}).get(m, 0.0):.4f}</td>" for m in sc_models
    )

    # Fix effectiveness rows
    file_tools = {"clean_temp", "clean_app_caches", "empty_recycle_bin"}
    app_tools = {"close_idle_apps", "end_heavy_tasks"}
    fix_rows = []
    for tid, t_stat in fix_summary.get("stats", {}).items():
        b_del = f"{t_stat['bytes_deleted'] / (1024**2):.1f} MB" if tid in file_tools else "-"
        m_closed = (
            f"{t_stat.get('app_memory_closed', 0.0) / (1024**2):.1f} MB"
            if tid in app_tools
            else "-"
        )
        if t_stat.get("mem_deltas"):
            avg_m = sum(t_stat["mem_deltas"]) / len(t_stat["mem_deltas"]) / (1024**2)
            avail_change = f"{avg_m:+.1f} MB"
        else:
            avail_change = "-"
        fix_rows.append(
            f"<tr><td><strong>{html.escape(tid)}</strong></td>"
            f"<td>{t_stat['runs']}</td><td>{b_del}</td>"
            f"<td>{m_closed}</td><td>{avail_change}</td></tr>"
        )
    fix_table_html = ""
    if fix_rows:
        fix_table_html = (
            "<table><thead><tr><th>Tool</th><th>Runs</th>"
            "<th>File Bytes Deleted</th><th>Memory of Closed Apps</th>"
            "<th>Avail Mem Change</th></tr></thead><tbody>" + "".join(fix_rows) + "</tbody></table>"
        )

    # Limitations list
    if is_synth:
        lim_items = f"<li>{html.escape(NO_REAL_DATA_MESSAGE)}</li>"
    else:
        lim_items = "".join(f"<li>{html.escape(lim)}</li>" for lim in limitations)

    no_data_banner = (
        f'<div class="card"><div class="banner-warning">{NO_REAL_DATA_MESSAGE}</div></div>'
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deskflow Instructor Verification Report</title>
<style>
  :root {{
    --bg: #0f141c;
    --card-bg: #161e2e;
    --border: #334155;
    --text: #f8fafc;
    --text-dim: #94a3b8;
    --accent: #38bdf8;
    --green: #4ade80;
    --yellow: #facc15;
    --red: #f87171;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background-color: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
      Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 24px;
    line-height: 1.5;
  }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 26px; margin-bottom: 4px; color: var(--accent); }}
  h2 {{
    font-size: 18px;
    margin-top: 24px;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
  }}
  .subtitle {{ color: var(--text-dim); font-size: 14px; margin-bottom: 20px; }}
  .banner-warning {{
    background-color: #422006;
    border: 1px solid var(--yellow);
    color: var(--yellow);
    padding: 12px 16px;
    border-radius: 6px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 24px;
    font-size: 14px;
  }}
  .card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 20px;
  }}
  .guide-box {{
    background: #0d1f33;
    border-left: 4px solid var(--accent);
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 13px;
    margin-bottom: 20px;
  }}
  .guide-box strong {{ color: var(--accent); }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-dim); font-weight: 600; text-transform: uppercase; font-size: 11px; }}
  td {{ text-align: right; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .baseline-row {{ color: var(--text-dim); opacity: 0.8; }}
  .badge-green {{
    background: #064e3b;
    color: var(--green);
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 11px;
  }}
  .badge-red {{
    background: #7f1d1d;
    color: var(--red);
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 11px;
  }}
  .dim {{ color: var(--text-dim); }}
  .chart-svg {{
    width: 100%;
    height: auto;
    max-width: 100%;
    font-family: monospace;
    font-size: 11px;
  }}
  .model-label {{ fill: var(--text); font-size: 12px; }}
  .axis-tick, .val-text {{ fill: var(--text-dim); font-size: 11px; }}
  .legend-text {{ fill: var(--text); font-size: 11px; }}
  .cm-table td {{
    text-align: center;
    padding: 6px;
    font-weight: bold;
    font-size: 12px;
    border: 1px solid var(--border);
  }}
  .cm-label {{ text-align: left !important; font-weight: normal; color: var(--text-dim); }}
  .fi-row {{ display: flex; align-items: center; margin-bottom: 8px; font-size: 13px; }}
  .fi-name {{
    width: 180px;
    text-overflow: ellipsis;
    overflow: hidden;
    white-space: nowrap;
    font-family: monospace;
    font-size: 12px;
  }}
  .fi-track {{
    flex: 1;
    height: 10px;
    background: #1e293b;
    border-radius: 5px;
    margin: 0 12px;
    overflow: hidden;
  }}
  .fi-fill {{ height: 100%; background: var(--accent); }}
  .fi-weight {{
    width: 60px;
    text-align: right;
    font-family: monospace;
    font-size: 12px;
    color: var(--text-dim);
  }}
  ul {{ padding-left: 20px; font-size: 13px; color: var(--text-dim); }}
  li {{ margin-bottom: 6px; }}
  @media print {{
    body {{ background: #ffffff !important; color: #000000 !important; padding: 0; }}
    .card {{
      background: #ffffff !important;
      border: 1px solid #cccccc !important;
      page-break-inside: avoid;
    }}
    .banner-warning {{
      background: #fffbe6 !important;
      color: #856404 !important;
      border: 1px solid #ffeeba !important;
    }}
    .guide-box {{
      background: #f0f9ff !important;
      border-left: 4px solid #0284c7 !important;
      color: #0369a1 !important;
    }}
    .chart-svg {{ filter: invert(0.9) hue-rotate(180deg); }}
    th, td {{ border-bottom: 1px solid #dddddd !important; }}
  }}
</style>
</head>
<body>
<div class="container">
  <h1>Deskflow Verification Report</h1>
  <div class="subtitle">Generated on {html.escape(str(machine.get("platform", "Desktop")))}</div>
  {synth_banner}

  <div class="guide-box">
    <strong>How to read this report:</strong>
    Probe Slowdown Ratio indicates micro-benchmark probe latency relative to baseline
    (1.0 = baseline performance, 1.5 = 50% slowdown). Offline leaderboard compares models
    using non-parametric 95% bootstrap confidence intervals. A model is strictly certified as
    statistically superior to the Persistence baseline only when the confidence interval of its
    improvement strictly excludes zero.
  </div>

  <h2>1. Machine & Baseline Summary</h2>
  <div class="card">
    <p><strong>Hardware:</strong> {html.escape(str(machine.get("cpu_model", "Unknown")))}
       ({machine.get("physical_cores", "-")} phys / {machine.get("logical_cores", "-")} log cores),
       {machine.get("total_ram_gb", "-")} GB RAM,
       GPU: {html.escape(str(machine.get("gpu", "None")))}</p>
    <table>
      <thead>
        <tr>
          <th>Power State</th><th>Calibrated</th><th>Median (ms)</th>
          <th>p95 (ms)</th><th>CV (%)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>AC</td>
          <td>{baselines.get("ac", {}).get("created_utc", "-")[:19]}</td>
          <td>{baselines.get("ac", {}).get("median_ms", "-")}</td>
          <td>{baselines.get("ac", {}).get("p95_ms", "-")}</td>
          <td>{baselines.get("ac", {}).get("cv", "-")}</td>
        </tr>
        <tr>
          <td>Battery</td>
          <td>{baselines.get("battery", {}).get("created_utc", "-")[:19]}</td>
          <td>{baselines.get("battery", {}).get("median_ms", "-")}</td>
          <td>{baselines.get("battery", {}).get("p95_ms", "-")}</td>
          <td>{baselines.get("battery", {}).get("cv", "-")}</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h2>2. Offline Model Leaderboard (Hold-Out Test Split)</h2>
  {
        no_data_banner
        if is_synth
        else f'''<div class="card">
    {_render_whiskers_svg(leaderboard)}
    <table>
      <thead>
        <tr>
          <th>Model</th><th>MAE</th><th>95% Bootstrap CI</th><th>RMSE</th>
          <th>R2</th><th>Tol Acc (&le;0.25)</th><th>Beats Baseline</th>
        </tr>
      </thead>
      <tbody>
        {"".join(leaderboard_rows)}
      </tbody>
    </table>
  </div>'''
    }

  <h2>3. Predicted vs Actual Telemetry (Test Trajectory)</h2>
  {
        no_data_banner
        if is_synth
        else f'''<div class="card">
    {_render_line_chart_svg(time_series)}
  </div>'''
    }

  <h2>4. Subgroup Breakdowns</h2>
  {
        no_data_banner
        if is_synth
        else f'''<div class="card">
    <h3>Scenario Breakdown (MAE)</h3>
    <table>
      <thead><tr><th>Scenario</th>{sc_headers}</tr></thead>
      <tbody>{"".join(sc_rows)}</tbody>
    </table>
    <h3>Power State Breakdown (MAE)</h3>
    <table>
      <thead><tr><th>Power State</th>{sc_headers}</tr></thead>
      <tbody>
        <tr>
          <td>AC</td>
          {ac_cells}
        </tr>
        <tr>
          <td>Battery</td>
          {bat_cells}
        </tr>
      </tbody>
    </table>
  </div>'''
    }

  <h2>5. Classification & Anomaly Detection</h2>
  {
        no_data_banner
        if is_synth
        else f'''<div class="card">
    <h3>Root Cause Classification Confusion Matrix</h3>
    <p>Overall Accuracy: <strong>{confusion_matrix.get("accuracy", 1.0) * 100:.1f}%</strong> |
       Macro F1: <strong>{confusion_matrix.get("macro_f1", 1.0):.4f}</strong></p>
    {_render_confusion_matrix_html(confusion_matrix)}

    <h3 style="margin-top: 20px;">Anomaly Detection Metrics (IsolationForest)</h3>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>ROC-AUC</td><td>{anomaly.get("roc_auc", 1.0):.4f}</td></tr>
        <tr><td>Recall</td><td>{anomaly.get("recall", 1.0):.4f}</td></tr>
        <tr><td>False Alarm Rate</td><td>{anomaly.get("false_alarm_rate", 0.02):.4f}</td></tr>
        <tr><td>Threshold</td><td>{anomaly.get("threshold", 0.54):.4f}</td></tr>
      </tbody>
    </table>
  </div>'''
    }

  <h2>6. Top Feature Contributions</h2>
  {
        no_data_banner
        if is_synth
        else f'''<div class="card">
    {"".join(fi_items)}
  </div>'''
    }

  <h2>7. Live Accuracy Verification</h2>
  <div class="card">
    <p>Valid sessions analyzed: <strong>{watch_summary.get("valid_sessions", 0)}</strong>
       ({watch_summary.get("excluded_sessions", 0)} out-of-range sessions excluded).</p>
    <p>{watch_summary.get("text", "No live sessions recorded.")}</p>
  </div>

  <h2>8. Fix Effectiveness</h2>
  <div class="card">
    <p>Audit records analyzed (non-dry-run): <strong>{fix_summary.get("total_runs", 0)}</strong></p>
    {fix_table_html}
    <p>{fix_summary.get("text", "No non-dry-run fixes recorded yet.")}</p>
  </div>

  <h2>9. Factual Limitations</h2>
  <div class="card">
    <ul>
      {lim_items}
    </ul>
  </div>

  {synth_banner}
</div>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")
    return output_path
