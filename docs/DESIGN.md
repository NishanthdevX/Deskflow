# Deskflow terminal design v2

## 1. Principles
Dense, dark-terminal, colour only with meaning. The look follows a mission-control dashboard:
bordered panels with bracketed titles, KPI cards, bars, sparklines, charts. Wording stays
plain and factual. Every rating word (Good, Fair, Poor, Safe, Caution, Pass) comes from a
numeric threshold defined in code. Never print text about agents, specs, compliance, phases
or marketing ("excellent", "cyber", "ultimate").

## 2. Palette (rich styles)
cyan #00E5FF bright_cyan      identity, ML predictions, benchmarks
green #00E676 spring_green2   good, passed, safe
yellow #FFEA00 yellow1        fair, warnings, recommendations, caution
red #FF1744 red1              poor, errors, failed
magenta #D500F9 magenta2      live accuracy, model tags
white bold                    primary values
grey #64748B grey50           secondary text, units, inactive borders
Colours never cycle decoratively. Border colour marks the domain: cyan = ML, green = health,
yellow = actions, magenta = live accuracy, grey = system facts.

## 3. Layout
Design width 100 columns, minimum 80, maximum 120. Below 100, card rows stack vertically.
Height-aware: live views must fit the terminal height (crop or compact, never duplicate frames).
Outer containers ROUNDED, tables SQUARE or HORIZONTALS. Alignment: text left, numbers right,
badges centred, bars left at fixed width.

## 4. Components
- Banner: block-letter DESKFLOW in cyan, home screen only, one muted subtitle line and the
  version. Plain title when width < 80, --plain, or non-UTF-8.
- Status strip: one row: OS, POWER, MODEL (name and DEMO flag if synthetic), BASELINE (age),
  duration.
- KPI card (4 lines): header in accent colour, hero value in bold white with muted context,
  pipe-separated sub-metrics, 14-character bar (full block and light shade) with percent.
  A bar appears only where a real fraction exists.
- Badges: SAFE (green), CAUTION (yellow), GOOD/FAIR/POOR from health thresholds, OK/WARN/FAIL.
- Bars: filled U+2588, track U+2591. Sparklines U+2581 to U+2588.
- Bar chart with interval: one row per model, bar for the metric, whisker for the 95%
  interval, baseline row marked and dimmed.
- Line chart: predicted versus actual over time, braille or block plot (plotext if its size
  and import time pass the ADR check, otherwise an in-house renderer).
- Heatmap: confusion matrix with shade blocks U+2591 U+2592 U+2593 U+2588 plus counts.
- Key-value block: used by info and doctor, label grey, value white, usage bars on the right.

## 5. Screens
HOME (loops until Q; each action returns here)
  banner
  [ THIS MACHINE ]  OS | CPU | RAM | power | baseline state        (one panel)
  [1] Full check   scan, fixes and report in one run
  [2] Quick scan   14 s health score and prediction
  [3] Fix Center   clean cache, free memory, close heavy tasks
  [4] Live watch   predictions versus reality, model comparison
  [5] System info  CPU, GPU, memory, disks
  [6] Report       graphs and model comparison
  [7] Prepare demo readiness checklist
  [Q] Quit

SCAN
  status strip
  KPI row:  [ HEALTH 87/100 Good ] [ NEXT 20 S 1.25x + sparkline ]
            [ LIKELY CAUSE tag + confidence ] [ UNUSUAL yes/no + score vs threshold ]
  split:    [ WHY ] bars of the top positive factors  |  [ RESOURCES ] CPU, memory, disk, swap
            usage bars with sparklines of the last 12 s
  [ HEAVY TASKS ] top 3 by CPU and top 3 by memory
  [ ACTIONS ] numbered fix tools with risk badge and measured potential; always shown, with
            "No action recommended" in the header when healthy. Footer: timings.

FIX CENTER
  inventory table: # | tool | risk badge | measured potential | what it changes
  prompt: choose numbers, "safe" for all safe tools, or Q. CAUTION tools get a preview and a
  separate confirmation. After running: [ BEFORE / AFTER ] table with delta column coloured by
  direction, plus freed bytes as measured.

INFO (fastfetch style)
  left: small block-letter logo or monitor glyph in cyan
  right: key-value lines (Host, OS, CPU with cores and clock, GPU with VRAM, Memory with usage
  bar, Disks with one bar each, Battery or power, Uptime, Display). Under 400 ms with cache.

WATCH
  header strip, [ NOW ] KPIs, line chart of predicted versus actual for the primary model,
  [ MODEL LEADERBOARD ] with MAE bars and the Persistence baseline row dimmed, last 5 resolved
  predictions. Fits terminal height. Times shown in local time.

REPORT
  [ MACHINE ] [ BASELINES by power state ] [ OFFLINE LEADERBOARD ] bar chart with intervals
  [ PREDICTED VS ACTUAL ] line chart  [ CONFUSION MATRIX ] heatmap  [ LIVE ACCURACY ]
  [ FIX EFFECTIVENESS ]  [ LIMITATIONS ]. A synthetic model shows the
  "SYNTHETIC DATA - NOT A RESULT" banner on every report section.

## 6. Accessibility and fallbacks
--plain: ASCII boxes and bars, no colour. NO_COLOR is honoured. Windows: UTF-8 stdout set at
startup. Non-UTF-8 or dumb terminals fall back to ASCII automatically. Recommend Windows
Terminal in docs. Output must stay readable in a pasted screenshot at 100 columns.

## 7. Performance rules
Lazy imports; UI code must not import sklearn or xgboost. Rendering a screen takes under 50 ms.
Live views refresh at 2 Hz at most. Banner and menu appear in under 1 second.

## 8. Tests
Golden text renders at widths 80, 100 and 120 in normal and --plain mode for every screen, no
exceptions, no emoji, --plain output pure ASCII, live views never exceed terminal height.
