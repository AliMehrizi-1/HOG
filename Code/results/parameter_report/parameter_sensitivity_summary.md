# Parameter sensitivity summary

> This analysis evaluates a separable 1-D Taylor model applied independently along x and y. It is not the joint full 2-D Taylor model.

## Scope and reporting hygiene

The report uses `parameter_analysis_results.csv` with matched clean references. Available seed(s): `42`. The present conclusions are descriptive for the tested image and seed=42. No uncertainty interval can be estimated from one seed.

All descriptor metrics use fixed Robust HOG with L2-Hys normalization and Huber weighting (delta=2.0, window=5). Runtime definition: Median of 3 noisy gradient-plus-HOG runs; per-configuration design-matrix setup and clean-reference computation excluded. Residual definition: Unweighted RMS Taylor-fit residual over both per-axis systems on the common analysis ROI; diagnostic, not correntropy loss. Residual and iteration count are diagnostics and are not components of the composite score.

The two sweeps are one-factor-at-a-time: the n sweep fixes h=1, and the h sweep fixes n=8. Therefore n=32 and h=4 must not be combined and described as an observed joint optimum. Gradient RMSE measures matched-clean magnitude stability, not absolute derivative accuracy.

## Effect of increasing n

Under salt-and-pepper noise, correntropy improved consistently as n increased from 16 to 32. At 5% noise, gradient RMSE decreased by 60.3% (2.2623 to 0.8991) and HOG relative error decreased by 19.7%, while runtime increased by 54.2%. At 10% noise the corresponding RMSE reduction was 65.3% and the runtime increase was 42.1%. Thus n=32 is the robustness-maximizing tested value, whereas n=16 is a more economical correntropy setting.

For squared loss, lower gradient RMSE did not always produce a more stable HOG descriptor. In particular, the descriptor metrics can flatten or worsen at the largest support. This is why the report keeps gradient RMSE, HOG error, and cosine as separate outcomes.

## Effect of increasing h

For the tested settings with n=8, increasing h improved all three matched-clean stability metrics. For correntropy, moving from h=3 to h=4 reduced RMSE by 21.6% at 5% noise and 24.0% at 10% noise. The corresponding HOG gains were smaller: cosine increased by 0.0049 and 0.0052. Hence h=4 is the best tested value for noise stability, while h=3 is a cautious alternative when localization and matrix conditioning are important. These data do not establish an unrestricted optimum beyond the tested boundary.

## Direct versus iterative squared loss

Across the noisy OFAT candidates, iterative squared loss changed gradient RMSE by a median improvement of 2.94% and changed HOG cosine by at most 0.0063, while requiring a median 3.9x the runtime of direct squared loss. Direct squared loss is therefore the clearer fast baseline; the iterative variant offers only a small stability difference for its additional computation in this experiment.

## Correntropy robustness-runtime trade-off

At n=16,h=1, correntropy reduced RMSE by 51.7% and HOG error by 54.4% at 5% noise, with a 53.2x runtime cost relative to direct squared loss. At 10% noise the reductions were 40.8% and 26.5%, with a 72.3x cost. The robustness improvement is substantial enough to justify correntropy when impulsive-noise stability is the priority, but not as an unqualified replacement when runtime dominates.

## Parameter recommendation

- Maximum measured robustness: correntropy with n=32,h=1.
- Cost-aware correntropy choice: n=16,h=1.
- Fast squared-loss baseline: direct squared loss.
- Spacing sweep at n=8: h=4 has the best tested stability; h=3 is the more conservative localization/conditioning compromise.

Using the requested global min-max composite and then averaging the two noisy-scenario scores equally, the leading physical candidate is Correntropy loss with n=32, h=1 (mean score 0.8661). This score is descriptive and candidate-set dependent, and it weights two correlated HOG measures for a combined 75% contribution.

## Main-paper figures

- `n_sweep_gradient_rmse.png`, `n_sweep_hog_error.png`, `n_sweep_cosine.png`, `n_sweep_runtime.png`
- `h_sweep_gradient_rmse.png`, `h_sweep_hog_error.png`, `h_sweep_cosine.png`, `h_sweep_runtime.png`
