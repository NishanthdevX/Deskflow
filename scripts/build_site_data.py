"""Generate docs/site/data/project-data.json from model artifacts (public-safe whitelist).
Usage: python scripts/build_site_data.py model_card.json report_data.json
"""

import json
import pathlib
import sys

if len(sys.argv) < 3:
    print("Usage: python scripts/build_site_data.py model_card.json report_data.json")
    sys.exit(1)

card_path = pathlib.Path(sys.argv[1])
rep_path = pathlib.Path(sys.argv[2])

card = json.loads(card_path.read_text(encoding="utf-8"))
rep = json.loads(rep_path.read_text(encoding="utf-8"))

out = {
    "status": "ok",
    "synthetic": card.get("synthetic", False),
    "dataset_sha256": card.get("dataset_sha256"),
    "train_rows": card.get("train_rows"),
    "test_rows": card.get("test_rows"),
    "sessions": card.get("sessions", card.get("sessions_count")),
    "machines": card.get("machines", card.get("machines_count")),
    "power_states": card.get("power_states"),
    "created": card.get("created", card.get("created_utc")),
    "git_commit": card.get("git_commit"),
    "version": card.get("version", card.get("deskflow_version")),
}

if card.get("synthetic") is False:  # truth contract: no metrics from a synthetic model
    if "regression" in rep and rep["regression"]:
        out["regression"] = rep["regression"]
    elif "leaderboard" in rep:
        out["regression"] = [
            {
                "name": row.get("model") or row.get("name"),
                "mae": row.get("mae"),
                "ci": row.get("mae_ci") or row.get("ci"),
                "baseline": row.get("is_baseline", row.get("baseline", False)),
            }
            for row in rep["leaderboard"]
        ]
    cls_metrics = card.get("test_metrics", {}).get("classification")
    out["classification"] = rep.get("classification") or cls_metrics
    anom_metrics = card.get("test_metrics", {}).get("anomaly")
    out["anomaly"] = rep.get("anomaly") or anom_metrics

# Destination targets: prioritize docs/site/data/project-data.json if in repo root
dest_site = pathlib.Path("docs/site/data/project-data.json")
written = []

if dest_site.parent.exists() or pathlib.Path("docs").exists():
    dest_site.parent.mkdir(parents=True, exist_ok=True)
    dest_site.write_text(json.dumps(out, indent=2), encoding="utf-8")
    written.append(str(dest_site))

# Also check data/project-data.json or project-data.json if running directly inside site folder
for p in [pathlib.Path("data/project-data.json"), pathlib.Path("project-data.json")]:
    if p.exists():
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        written.append(str(p))

if not written:
    dest_site.parent.mkdir(parents=True, exist_ok=True)
    dest_site.write_text(json.dumps(out, indent=2), encoding="utf-8")
    written.append(str(dest_site))

print(f"wrote {', '.join(written)}")
