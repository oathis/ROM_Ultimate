# mini_case sample raw data

This folder contains a tiny synthetic raw dataset for fast ROM flow checks.

- files: `xresult-*.csv` (8 time steps)
- nodes per file: 12
- required columns for preprocess are included:
  - `x-coordinate`, `y-coordinate`, `z-coordinate`
  - `x-velocity`, `y-velocity`, `z-velocity`
  - `temperature`

Use this folder as `--raw-dir` in split/preprocess commands.
