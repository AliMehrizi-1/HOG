# Final parameter-analysis package

This experiment evaluates a separable 1-D Taylor model applied independently along x and y. It is not the joint full 2-D Taylor model.

separable 1-D Taylor along x/y; matched-clean metrics measure noise stability, not absolute derivative accuracy.

## Contents

- `n_visual_comparison.png`: Direct and Correntropy Robust-HOG panels for n = 4, 8, 16, 32 at h = 1 and SP = 10%.
- `h_visual_comparison.png`: Direct and Correntropy Robust-HOG panels for h = 1, 2, 3, 4 at n = 8 and SP = 10%.
- `joint_correntropy_sweep.csv`: the 12 measured sparse joint rows.
- `joint_correntropy_selection.csv`: averaged scores and the fastest, balanced, and maximum-weighted-robustness labels.
- `parameter_summary.tex`: compact numerical table.
- `runtime_scaling_trials.csv` and `runtime_scaling_summary.csv`: measured timing provenance.
- `runtime_scaling_2d.png`: shared-axis, log-runtime line plots.
- `parameter_analysis_subsection.tex`: manuscript-ready subsection with no unresolved numerical placeholders.
- `package_metadata.json`: experiment and environment metadata.

## Selected sparse-joint configurations

- Fastest: n = 16, h = 4.
- Balanced: n = 16, h = 4.
- Maximum weighted robustness: n = 32, h = 1.

Selections are restricted to the six measured correntropy pairs. Raw SP 5% and SP 10% metrics are first averaged per configuration; the six configuration means are then min-max benefit scaled. Weighted robustness uses relative weights 0.20 RMSE, 0.45 HOG relative error, and 0.30 cosine. Balanced is the harmonic mean of weighted robustness and min-max log-runtime benefit.

Runtime scaling reports implementation-level pipeline time. At h = 1, the native solve area is (S - n)^2 before every result is cropped to the common (S - 32)^2 ROI; timings are not normalized per output pixel. The IQR from 3 repeats is descriptive, not a confidence interval. Image loading/resizing, noise generation, and design-matrix construction are outside the timer.

## Reproduce

Run from the repository root:

```powershell
python -B final_parameter_analysis.py
```

Existing complete benchmark CSVs are reused only when the recorded image hash, solver settings, design, repeats, and core library versions match. Add `--force-joint` or `--force-runtime` to repeat the affected measurements.

Joint runtimes use the median of 3 runs. Runtime scaling uses 3 trials per point.

No legacy 3-D image-size runtime dataset was present, so the 2-D runtime plot is based on newly measured trials rather than values inferred from the one-size sensitivity CSV.

The original one-factor sensitivity CSV does not contain a seed column; its seed 42 provenance is recorded in `parameter_analysis_config.json` and the generating source. The joint and runtime CSVs record seed 42 explicitly.
