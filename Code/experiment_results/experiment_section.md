# Experimental evaluation

## Protocol

We evaluated six grayscale images (`cameraman`, `lena`, `home`, `girl`,
`flower`, and `clock`). Images were resized while preserving aspect ratio so
that the longest side was 256 pixels. Three deterministic noise realizations
(seeds 11, 29, and 47) were used for every image and condition. The same noisy
realization was supplied to all compared methods, giving 18 paired
image-realization trials per noise level.

Salt-and-pepper noise was evaluated at exact nominal densities of 1%, 5%, and
10%. Additive white Gaussian noise was evaluated at target contrast SNR values
of 10, 20, and 30 dB. Contrast SNR was defined using the mean-removed image
power, and Gaussian samples were not clipped after corruption.

Three HOG pipelines were compared:

1. **A — Sobel + Standard HOG**
2. **B — Taylor/Maclaurin + Standard HOG**
3. **C — Taylor/Maclaurin + Robust HOG**

All three pipelines used the same valid image region, cell size, orientation
bins, angular interpolation, block geometry, and block-normalization code.
Consequently, A versus B isolates the gradient estimator, while B versus C
isolates robust magnitude weighting. The robust method multiplied each
gradient magnitude by a Huber weight derived from its deviation from a local
component-wise median gradient. Gradient orientation was never modified.
Hyperparameters were fixed for all test conditions (`delta=2`, window size 5).
Both L2 and L2-Hys block normalization were evaluated.

The gradient stability metric was

```text
sqrt(mean((dx_noisy-dx_clean)^2 + (dy_noisy-dy_clean)^2)).
```

This measures sensitivity to noise relative to each estimator's own clean
output; it is not an absolute derivative error against analytical ground
truth. HOG stability was measured on the canonical overlapping-block
descriptor using cosine similarity and relative L2 error:

```text
cosine = dot(H_noisy, H_clean) / (||H_noisy|| ||H_clean||)
relative error = ||H_noisy - H_clean|| / ||H_clean||
```

To ensure that robust suppression did not produce an artificially stable but
collapsed representation, robust descriptors were also compared with the
clean Taylor + Standard HOG descriptor as a common reference. Noise seeds were
averaged within each image before computing plotted 95% intervals across
images.

## Results

Taylor/Maclaurin gradient estimation had lower stability RMSE than Sobel in all
108 paired trials. The mean Sobel-to-Taylor RMSE ratios were 6.07x, 4.12x, and
3.34x at salt-and-pepper densities of 1%, 5%, and 10%, respectively. Under
Gaussian noise, the corresponding ratios were 2.16x, 1.80x, and 1.30x at 10,
20, and 30 dB.

Under salt-and-pepper noise, Taylor + Robust HOG improved on Taylor + Standard
HOG in every paired trial when each method was compared with its own clean
descriptor. With L2-Hys normalization, the common-reference cosine improvement
was positive in 98.1% of trials. Averaged over the three impulse densities,
Robust HOG increased cosine similarity by 0.0380 and reduced relative error by
0.1012 compared with Taylor + Standard HOG. At 10% density, for example,
L2-Hys cosine similarity increased from 0.7858 to 0.8322 and relative error
decreased from 0.6928 to 0.6080.

The robust weighting did not provide a consistent advantage under Gaussian
noise. Its mean changes relative to Taylor + Standard HOG were close to zero
and slightly negative under the common-reference analysis. This is consistent
with the intended role of bounded Huber weighting: it suppresses sparse
high-energy impulses, whereas dense Gaussian perturbations are better handled
by the Taylor estimator itself and block normalization. The supported claim is
therefore that the proposed Robust HOG improves robustness to impulsive noise,
not that it universally dominates Standard HOG for every noise distribution.

## Limitations and paper-use guidance

These six-image results are a reproducible pilot study, not a final
dataset-level benchmark. A paper submission should freeze the robust
hyperparameters using only a training/validation split and repeat the same
script on the complete BSDS test split with more noise seeds. Because the
Taylor estimator has wider spatial support than a 3x3 Sobel operator, a
matched-support or equally smoothed Sobel control is needed before attributing
all improvements solely to the Taylor model. Absolute gradient-accuracy claims
require synthetic functions with known analytical derivatives. Finally,
descriptor robustness should ideally be complemented by a downstream
classification or detection experiment.
