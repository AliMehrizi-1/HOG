# Parameter sensitivity reporting pipeline

This analysis evaluates a separable 1-D Taylor model applied independently along x and y. It is not the joint full 2-D Taylor model.

Run from the project root:

```powershell
python -B parameter_sensitivity_report.py
```

Custom input/output example:

```powershell
python -B parameter_sensitivity_report.py --input-csv "D:\Papers\HOG\robust-HOG\parameter_analysis_results.csv" --output-dir "results\parameter_report"
```

The script reads the existing experiment CSV. If the default CSV is missing, it invokes `parameter_analysis_pipeline.py` with `polynomial_order=3` and seed 42 before reporting. Repeat `--input-csv` to add future seed-specific result files; every file in a multi-file run must contain an explicit `seed` column. Plotted and ranked metrics are then averaged by evaluated configuration after complete-design validation.

All descriptor metrics use fixed Robust HOG with L2-Hys normalization and Huber weighting (delta=2.0, window=5). Runtime definition: Median of 3 noisy gradient-plus-HOG runs; per-configuration design-matrix setup and clean-reference computation excluded. Residual definition: Unweighted RMS Taylor-fit residual over both per-axis systems on the common analysis ROI; diagnostic, not correntropy loss. Residual and iteration count are diagnostics and are not components of the composite score.

Outputs include eight 300-dpi main-paper figures, canonicalized source data (including no-noise regression rows), best-parameter tables, the noisy-only composite ranking, and Markdown/LaTeX conclusions.

The union ranking contains only seven evaluated OFAT pairs, not a 4-by-4 factorial grid. Runtime figures use a logarithmic y-axis. No-noise rows remain in `canonical_parameter_results.csv` but are excluded from all noisy-condition rankings.
