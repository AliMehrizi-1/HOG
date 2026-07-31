# Final Sobel comparison experiments

The experiments reuse `AxisTaylorSolver`, `_estimate_separable_gradient`,
`sobel_gradient`, and `describe_hog` from the existing project. Fixed method
configurations come from the completed parameter analyses; no displayed or
test image is used for tuning.

Run all experiments from the repository root:

```powershell
python run_sobel_comparison.py --config configs/sobel_comparison.json
```

Or run them separately:

```powershell
python qualitative_sobel_comparison.py --config configs/sobel_comparison.json
python quantitative_sobel_comparison.py --config configs/sobel_comparison.json
python multicue_edge_evaluation.py --config configs/sobel_comparison.json
```

Display CLI help with `python <script>.py --help`.

Outputs are written to `results/sobel_comparison/`. The quantitative script
evaluates every path listed in `test_images`, using seed 42 plus the
zero-based image index and reusing each noisy image across all four methods.
All methods use robust HOG and the same 32-pixel ROI.

MultiCue is optional. To enable it, set `multicue.root` in the config and
provide paired files:

```text
datasets/MultiCue/
  images/
    <image-id>.jpg
  groundTruth/
    <image-id>.mat
```

Without both images and human boundary annotations, the script writes
`multicue_metadata.json` with `status: SKIPPED` and does not fabricate CSV,
figure, or LaTeX results.
