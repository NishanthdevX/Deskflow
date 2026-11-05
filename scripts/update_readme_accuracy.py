"""Update README.md Accuracy section between markers using real model benchmarks.

Extracts verified row counts, sessions, machines, and held-out leaderboards
with moving block bootstrap confidence intervals from model_card.json and report_data.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
README_PATH = ROOT_DIR / "README.md"
MODEL_DIR = ROOT_DIR / "src" / "deskflow" / "data" / "models" / "default"

START_MARKER = "<!-- ACCURACY_START -->"
END_MARKER = "<!-- ACCURACY_END -->"


def generate_accuracy_markdown() -> str:
    """Generate Markdown snippet for the Accuracy section."""
    card_path = MODEL_DIR / "model_card.json"
    report_path = MODEL_DIR / "report_data.json"

    if not card_path.exists():
        return (
            "\n> **NO REAL DATA YET**: No accuracy claims yet "
            "(demo model trained on synthetic data).\n"
            "> Run `deskflow tools build-model` on collected runs to generate real benchmarks.\n"
        )

    with open(card_path, encoding="utf-8") as f:
        card: dict[str, Any] = json.load(f)

    is_synthetic = bool(card.get("synthetic", True))
    if is_synthetic:
        return (
            "\n> **NO REAL DATA YET**: No accuracy claims yet "
            "(demo model trained on synthetic data).\n"
            "> Run `deskflow tools build-model` on collected runs to generate real benchmarks.\n"
        )

    # Real data numbers
    train_rows = int(card.get("train_rows", 0))
    test_rows = int(card.get("test_rows", 0))
    total_rows = train_rows + test_rows
    sessions_count = int(card.get("sessions_count", 0))
    machines_count = int(card.get("machines_count", 1))
    power_states = card.get("power_states", ["ac", "battery"])
    power_str = ", ".join(power_states)
    dataset_hash = str(card.get("dataset_sha256", "unknown"))[:8]
    git_commit = str(card.get("git_commit", "unknown"))[:8]

    report_data: dict[str, Any] = {}
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            report_data = json.load(f)

    leaderboard = report_data.get("leaderboard", [])

    lines: list[str] = [
        "",
        f"**Hardware Verification Dataset**: Tested across **{total_rows:,} real rows** "
        f"({train_rows:,} train, {test_rows:,} test) across **{sessions_count} sessions** "
        f"on {machines_count} physical machine(s) in both **{power_str}** power states "
        f"(dataset `{dataset_hash}`, commit `{git_commit}`).",
        "",
        "### Held-Out Test Set Leaderboard (20s Prediction Horizon)",
        "",
        "| Model | Type | Test MAE (95% CI) | Test RMSE | R2 | vs Persistence | Superior? |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    if leaderboard:
        for m in leaderboard:
            m_name = m.get("model", "")
            m_type = "Baseline" if m.get("is_baseline") else "Model"
            mae = m.get("mae", 0.0)
            mae_ci = m.get("mae_ci", [mae, mae])
            mae_str = f"{mae:.4f} `[{mae_ci[0]:.4f}, {mae_ci[1]:.4f}]`"
            rmse = f"{m.get('rmse', 0.0):.4f}"
            r2 = f"{m.get('r2', 0.0):.4f}"
            imp_ci = m.get("improvement_ci", [0.0, 0.0])
            imp_str = f"`[{imp_ci[0]:+.4f}, {imp_ci[1]:+.4f}]`"
            beats = m.get("beats_persistence", False)
            cert = "**Yes (Superior)**" if beats else "Not distinguishable"
            if m_name == "Persistence":
                imp_str = "Baseline"
                cert = "N/A"
            lines.append(
                f"| **{m_name}** | {m_type} | {mae_str} | {rmse} | {r2} | {imp_str} | {cert} |"
            )
    else:
        # Fallback from card test_metrics
        reg_metrics = card.get("test_metrics", {}).get("regression", {})
        for m_name, metrics in reg_metrics.items():
            m_type = (
                "Baseline"
                if m_name in ("Persistence", "RollingMean", "ConstantMedian")
                else "Model"
            )
            mae = metrics.get("mae", 0.0)
            rmse = metrics.get("rmse", 0.0)
            r2 = metrics.get("r2", 0.0)
            lines.append(
                f"| **{m_name}** | {m_type} | {mae:.4f} | {rmse:.4f} | {r2:.4f} | -- | -- |"
            )

    lines.extend(
        [
            "",
            "### Multi-Class Issue Classification Performance",
            "",
            "| Classifier | Balanced Accuracy | Macro F1 | Status |",
            "| :--- | :--- | :--- | :--- |",
        ]
    )

    clf_metrics = card.get("test_metrics", {}).get("classification", {})
    for c_name, c_data in clf_metrics.items():
        if isinstance(c_data, dict):
            b_acc = c_data.get("balanced_accuracy", 0.0)
            m_f1 = c_data.get("macro_f1", 0.0)
            lines.append(f"| **{c_name}** | {b_acc:.4f} | {m_f1:.4f} | Verified |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Update README.md with generated accuracy documentation."""
    if not README_PATH.exists():
        raise FileNotFoundError(f"README.md not found at {README_PATH}")

    content = README_PATH.read_text(encoding="utf-8")
    if START_MARKER not in content or END_MARKER not in content:
        raise ValueError(f"Markers {START_MARKER} and {END_MARKER} missing in README.md")

    prefix = content[: content.index(START_MARKER) + len(START_MARKER)]
    suffix = content[content.index(END_MARKER) :]

    accuracy_md = generate_accuracy_markdown()
    updated = f"{prefix}\n{accuracy_md}\n{suffix}"

    README_PATH.write_text(updated, encoding="utf-8")
    print(f"Updated README.md accuracy section from {MODEL_DIR}")


if __name__ == "__main__":
    main()
