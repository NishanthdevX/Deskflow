# Deskflow Console Design Specification (v4)

Deskflow is an agentic, dense, keyboard-driven terminal console utility. It delivers actionable hardware responsiveness diagnostics and machine learning slowdown predictions.
Zero emojis are permitted in code, console output, documentation, or commit messages. Color carries operational meaning only; typography and box borders provide structure.

---

## 1. Design Philosophy

1. **3-Second Verdict**: Every diagnostic check leads with an unambiguous, human-readable summary sentence stating machine status, 20-second predicted slowdown ratio, and immediate actionable remediation.
2. **Never Wait on a Dead Bar**: Telemetry measurement runs a live dashboard featuring real-time probe latency sparklines, resource utilization gauges, and prediction countdowns evaluated live against ground truth.
3. **Alternate Screen vs. Scrollback**: Long-lived interactive interfaces (Home menu, Live dashboards, Pipeline tour) run full-screen. Result screens print directly to the terminal scrollback buffer so users can scroll, review, and copy outputs.
4. **Keyboard-First Navigation**: Navigation supports Arrow keys, Space, Enter, hotkeys `[1-6]`, `[F]` Fix Center, `[W]` Watch, `[?]` Glossary, and `[Q]` Quit, with non-blocking key reads and typed-input fallback.
5. **Single Source of Truth (`ModelFacts`)**: All model performance metrics, dataset row counts, sessions, and training provenance derive from `ModelFacts`. Hard-coded numbers across screens are strictly forbidden.
6. **Strict Safety Protocol**: System processes (`Memory Compression`, `csrss.exe`, `MsMpEng.exe`) and active developer IDEs are protected and excluded from termination candidates. Cleanable caches strictly enforce allowlist boundaries.

---

## 2. Responsive Breakpoint Matrix

Layouts dynamically inspect terminal dimensions (`console.size.width`, `console.size.height`) via `deskflow.ui.layout`:

### Width Breakpoints
- **Wide ($\ge 120$ cols)**:
  - Home: 2-column layout for `THIS MACHINE` and `MODEL FACTS`; 2-column action menu.
  - Check: 4 KPI cards side-by-side (`HEALTH`, `NEXT 20 S`, `LIKELY CAUSE`, `ANOMALY CHECK`).
  - Info: 3 side-by-side panels (`HARDWARE`, `MEMORY AND DISKS`, `POWER AND DISPLAY`).
- **Normal (100–119 cols)**:
  - Cards and panels maintain horizontal layout with compact cell padding.
- **Compact (80–99 cols)**:
  - 4 KPI cards fold into a $2 \times 2$ grid.
  - Info panels arrange into $2 + 1$ or vertical stack.
  - Model comparison charts scale bar widths proportionally.
- **Stacked ($< 80$ cols)**:
  - All cards, panels, and tables collapse into single-column vertical stacks.
  - Top banner drops ASCII art and renders as a single-line plain title.

### Height Tiers
- **Tall ($\ge 34$ rows)**: Renders full 6-row ASCII art banner with serif block legs (`╚═════╝ ...`).
- **Standard (26–33 rows)**: Renders compact 2-row wordmark banner.
- **Short ($< 26$ rows)**: Renders single-line plain title (`DESKFLOW v0.2.0`) to preserve vertical space.

---

## 3. Official Theme Palette (`DESKFLOW_THEME`)

Deskflow implements custom theme styles defined in `deskflow.ui.theme`:

| Style Name | Color / Format | Purpose |
| :--- | :--- | :--- |
| `df.title` | `bold cyan` | Primary headers, wordmarks, brand titles |
| `df.good` | `bold green` | Healthy status, optimal scores ($\ge 85$), negative latency delta |
| `df.warn` | `bold yellow` | Degraded status, moderate load (60–84), warnings |
| `df.bad` | `bold red` | Critical status, poor scores ($< 60$), severe bottlenecks |
| `df.accent` | `cyan` | Stage highlights, current selections, active markers |
| `df.muted` | `dim` | Explanatory labels, disabled options, table hints |
| `df.value` | `bold white` | Quantitative telemetry metrics, timings, percentages |
| `df.border.good` | `green` | Card border when health is optimal |
| `df.border.warn` | `yellow` | Card border when health is degraded |
| `df.border.bad` | `red` | Card border when health is critical |
| `df.border.dim` | `bright_black` | Default neutral panel borders |

### Color Blending
- `blend_health_color(score)`: Smoothly interpolates `red` $\to$ `yellow` $\to$ `green` across 0–100 health range.
- `blend_load_color(load)`: Smoothly interpolates `green` $\to$ `yellow` $\to$ `red` across 0–100 load percentage.

---

## 4. Keyboard Navigation & Screen Lifecycle

Interactive screens inherit from `deskflow.ui.app.Screen`:
- `render() -> RenderableType`: Generates the Rich renderable tree for the current frame.
- `handle_key(key: str) -> Optional[str]`: Processes keyboard input, returning state transition signals (`"quit"`, `"refresh"`, or screen name).
- `run(console)`: Standard event loop running at 10 Hz with non-blocking key reads via `deskflow.platform.base.read_key(timeout=0.1)`.
  - **Windows**: Implemented using `msvcrt.kbhit()` and `msvcrt.getwch()`, mapping prefix sequences (`\x00`, `\xe0`) to `up`, `down`, `left`, `right`.
  - **macOS / Linux**: Implemented using `termios` raw mode and `select.select()` on `sys.stdin`.
  - **Fallback**: Graceful fallback to `rich.prompt.Prompt` when standard input is not an interactive TTY.

---

## 5. Screen Layouts & Wireframes

### 5.1 Home Screen Wireframe
```text
  ██████╗  ███████╗ ███████╗ ██╗  ██╗ ███████╗ ██╗       ██████╗  ██╗    ██╗
  ██╔══██╗ ██╔════╝ ██╔════╝ ██║ ██╔╝ ██╔════╝ ██║      ██╔═══██╗ ██║    ██║
  ██║  ██║ █████╗   ███████╗ █████═╝  █████╗   ██║      ██║   ██║ ██║ █╗ ██║
  ██║  ██║ ██╔══╝   ╚════██║ ██╔═██╗  ██╔══╝   ██║      ██║   ██║ ██║███╗██║
  ██████╔╝ ███████╗ ███████║ ██║  ██╗ ██║      ███████╗ ╚██████╔╝ ╚███╔███╔╝
  ╚═════╝  ╚══════╝ ╚══════╝ ╚═╝  ╚═╝ ╚═╝      ╚══════╝  ╚═════╝   ╚══╝╚══╝ 
  DESKFLOW v0.2.0 - Device health scan with ML slowdown prediction

┌────────────── [ THIS MACHINE ] ───────────────┐ ┌─────────────── [ MODEL FACTS ] ────────────────┐
│ Host / OS: Anonymous1122 (Windows 11)         │ │ Status: REAL HARDWARE MODEL                    │
│ CPU: AMD Ryzen 7 6800HS (8C/16T @ 3200 MHz)   │ │ Dataset: 1,457 train, 500 held-out test rows   │
│ GPU: NVIDIA GeForce RTX 3060 Laptop GPU       │ │ (1,957 total) across 6 sessions on 1 machine   │
│ RAM: 15.2 GB | Power: AC Wall Power           │ │ Horizon: 20s ahead slowdown prediction         │
│ Power: AC Wall Power                          │ │ Baselines: AC (fresh, 2h ago) | Battery (fresh)│
└───────────────────────────────────────────────┘ └────────────────────────────────────────────────┘

[1] Battery check     about 75s  health,          [4] Fix Center        clean cache, memory, idle   
prediction, accuracy                              tasks                                             
[2] Plugged-in check  about 75s  same, on AC      [5] How it works      animated pipeline &         
wall power                                        architecture                                      
[3] Full scan         about 10m  automated        [6] System info       fastfetch CPU, GPU, memory, 
hardware stress suite                             disks                                             

[1-6] Select tool                              [?] Glossary                      [Q] Quit           
```

### 5.2 Check Dashboard Wireframe
```text
OS: Windows 11       POWER: AC      MODEL: RandomForest        BASELINE: 2h ago       SCAN: 14.2s   
────────────────────────────────────────────────────────────────────────────────────────────────────
Your laptop is running normally. Predicted slowdown 1.04x. Nothing urgent; 279 MB of cleanable files
found.

┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐ ┌───────────────────────┐
│ [HEALTH]             │ │ [NEXT 20 S]          │ │ [LIKELY CAUSE]       │ │ [ANOMALY CHECK]       │
│ 92/100  Optimal      │ │ 1.04x  vs baseline   │ │ Healthy  96% conf    │ │ NORMAL  0.21 score    │
│ 100 = Baseline |     │ │ Horizon 20s | 104%   │ │ health: 96% | cpu:   │ │ IsolationForest |     │
│ Optimal status       │ │ baseline             │ │ 2% | load: 1%        │ │ Thresh 0.60           │
│ █████████████░ 92%   │ │ █████░░░░░░░░░ 35%   │ │ █████████████░ 96%   │ │ █████░░░░░░░░░ 35%    │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘ └───────────────────────┘

┌─────────────── [ WHY ] ────────────────┐ ┌──────────────────── [ RESOURCES ] ────────────────────┐
│ Primary: Healthy background state      │ │ CPU 14%  ▂▃▅▃▂  | RAM 61% (9.3/15.2 GB) ▅▅▅▅▅▅         │
│ Factor 1: CPU headroom at 86%          │ │ Disk C: 142 GB free (78% used) █▉                     │
│ Factor 2: Available memory 5.9 GB      │ │ Read: 0.1 MB/s | Write: 1.4 MB/s                      │
│ Factor 3: Zero swap file pressure      │ │ Probe Latency: 2.1 ms  ▂  | Ctx Switch: 1,420/s        │
└────────────────────────────────────────┘ └────────────────────────────────────────────────────────┘

┌────────────────────────────────────── [ ACTIONS ] ───────────────────────────────────────────────┐
│ Cleanable Caches: 279 MB potential across 3 tools (Run Fix Center to clean)                      │
│ Heavy Tasks: None (all processes within normal operational bounds)                               │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────── [ MODEL COMPARISON ] ─────────────────────────────────────────┐
│ HELD-OUT TEST SPLIT (500 rows across 1 session on 1 machine)                                     │
│   MODEL               MAE     95% CI          BASELINE        ACCURACY COMPARISON                │
│   Persistence         0.0246  [0.021, 0.028]  [BASELINE]      ●──────────────────────────        │
│   RandomForest        0.0011  [0.000, 0.002]                  ●                                  │
│   XGBoost             0.0008  [0.000, 0.001]                  ●                                  │
│                                                                                                  │
│ LIVE PREDICTIONS (THIS CHECK)                                                                    │
│   RandomForest: evaluated 3 live prediction pairs | Live MAE: 0.46 (transitions between burst/idle)│
│                                                                                                  │
│ Offline: held-out steady-state test set. Live: dynamic non-stationary desktop workloads.         │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
Timings: preflight 0.3s | scan 14.2s | inference 31ms | total 14.5s
[Enter] Main Menu        [F] Fix Center        [W] Watch Mode        [Q] Quit
```

### 5.3 Fix Center Wireframe
```text
┌────────────────────────────────────── [ FIX CENTER ] ────────────────────────────────────────────┐
│ Safe remediation actions for reclaimable disk caches and background tasks                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

  #   TOOL                ESTIMATED POTENTIAL    DESCRIPTION                                        
  1   User Temp Cache     245.2 MB               Temporary application scratch files                
  2   Crash Dumps         33.8 MB                Completed application crash minidumps              
  3   Windows Delivery    0 B (nothing to do)    Cached Windows Update peer-to-peer delivery chunks 

Total potential space reclaimable: 279.0 MB

[1-3] Toggle tool selection    [A] Apply remediation    [D] Dry run audit    [Q] Return to menu     
```

### 5.4 System Info Dashboard Wireframe
```text
┌────────────── [ HARDWARE ] ──────────────┐ ┌────────── [ MEMORY AND DISKS ] ──────────┐ ┌─────────── [ POWER AND DISPLAY ] ────────────┐
│ Host / OS   Anonymous1122 (Windows 11)   │ │ Physical RAM   15.2 GB total (9.3 GB used)│ │ AC Power       Connected (Wall Power)        │
│ CPU         AMD Ryzen 7 6800HS with      │ │ Available RAM  5.9 GB free headroom       │ │ Battery        Not Discharging               │
│             Radeon Graphics (8C/16T @    │ │ Swap / Page    2.4 GB used (14%)          │ │ Display        1920x1080 (Primary)           │
│             3200 MHz)                    │ │ Disk C:        142.1 GB free / 476.0 GB   │ │ Uptime         1h 14m                        │
│ GPU         NVIDIA GeForce RTX 3060      │ │                total                      │ │ Version        v0.2.0                        │
│             Laptop GPU (6.0 GB VRAM)     │ │ Disk Health    OK (SMART optimal)         │ │                                              │
└──────────────────────────────────────────┘ └───────────────────────────────────────────┘ └──────────────────────────────────────────────┘
```

---

## 6. Audit Trail & Verification Requirements

1. **Deterministic Metrics**: Metric outputs must be strictly tied to `ModelFacts`. No arbitrary manual constants in display code.
2. **Safety Exclusions**:
   - `_WINDOWS_PROTECTED_PROCESSES` and `_WINDOWS_NEVER_OFFER` block any attempt to kill system processes (`csrss.exe`, `Memory Compression`, `MsMpEng.exe`).
   - File deletion tools execute within canonical allowlisted scratch directories only.
3. **SVG & Text Export**: Every screen redesign exports matching `.svg` and `.txt` snapshots into `docs/screens/` via `scripts/update_screens.py`.
