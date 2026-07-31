# Robust HOG experiment summary

Values are means over all images and paired seeds. Gradient RMSE is method-specific noise stability, not absolute derivative accuracy.

| Noise | Norm | Sobel RMSE | Taylor RMSE | Gain | A cosine | B cosine | C cosine | A rel.err | B rel.err | C rel.err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S&P 1% | L2 | 9.2726 | 1.5462 | 6.07x | 0.8397 | 0.9206 | 0.9401 | 0.5864 | 0.3576 | 0.2575 |
| S&P 1% | L2-Hys | 9.2726 | 1.5462 | 6.07x | 0.8510 | 0.9243 | 0.9383 | 0.5605 | 0.3373 | 0.2600 |
| S&P 5% | L2 | 20.5328 | 4.9933 | 4.12x | 0.7418 | 0.8346 | 0.9002 | 0.7681 | 0.6025 | 0.4391 |
| S&P 5% | L2-Hys | 20.5328 | 4.9933 | 4.12x | 0.7578 | 0.8482 | 0.9019 | 0.7410 | 0.5717 | 0.4302 |
| S&P 10% | L2 | 28.8251 | 8.6349 | 3.34x | 0.6850 | 0.7732 | 0.8253 | 0.8480 | 0.7177 | 0.6253 |
| S&P 10% | L2-Hys | 28.8251 | 8.6349 | 3.34x | 0.7210 | 0.7858 | 0.8322 | 0.7959 | 0.6928 | 0.6080 |
| AWGN 10 dB | L2 | 11.2514 | 5.1979 | 2.16x | 0.7689 | 0.7926 | 0.7929 | 0.7248 | 0.6847 | 0.6837 |
| AWGN 10 dB | L2-Hys | 11.2514 | 5.1979 | 2.16x | 0.7858 | 0.8030 | 0.8043 | 0.6948 | 0.6621 | 0.6589 |
| AWGN 20 dB | L2 | 3.5580 | 1.9812 | 1.80x | 0.8576 | 0.8637 | 0.8623 | 0.5534 | 0.5360 | 0.5393 |
| AWGN 20 dB | L2-Hys | 3.5580 | 1.9812 | 1.80x | 0.8649 | 0.8702 | 0.8694 | 0.5348 | 0.5167 | 0.5183 |
| AWGN 30 dB | L2 | 1.1251 | 0.8699 | 1.30x | 0.8979 | 0.9044 | 0.9032 | 0.4354 | 0.4128 | 0.4179 |
| AWGN 30 dB | L2-Hys | 1.1251 | 0.8699 | 1.30x | 0.9033 | 0.9068 | 0.9057 | 0.4190 | 0.4012 | 0.4051 |

## Paired win rates

- Taylor lower-gradient-RMSE win rate: 100.0% (n=108).
- saltpepper_L2: Robust HOG own-clean cosine win rate 100.0%; common-reference cosine win rate 90.7% (n=54).
- saltpepper_L2-Hys: Robust HOG own-clean cosine win rate 100.0%; common-reference cosine win rate 98.1% (n=54).
- gaussian_L2: Robust HOG own-clean cosine win rate 25.9%; common-reference cosine win rate 18.5% (n=54).
- gaussian_L2-Hys: Robust HOG own-clean cosine win rate 35.2%; common-reference cosine win rate 11.1% (n=54).
