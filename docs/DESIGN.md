# Deskflow terminal design

Deskflow uses dense, dark-terminal-friendly output. Colour carries status only: cyan for ML,
green for good health, yellow for warnings and recommendations, red for poor/error, magenta for
prediction tags, and grey for system facts. Rating words come only from documented thresholds.

Panels use uppercase bracketed titles, rounded outer borders, and square or horizontal tables.
Scan shows a five-field status strip, four KPI cards, factors/categories, recommendations, and a
timing footer. Fix shows actions and before/after deltas. Doctor uses a capability table.

Unicode block bars and sparklines are used only when UTF-8 output is available. `--plain`,
`NO_COLOR`, dumb terminals, and non-UTF-8 streams use ASCII bars and boxes with no colour.
At widths below 100, panel groups stack. The home screen alone may show the cyan Deskflow banner;
it becomes a plain title below 80 columns or in plain mode.
