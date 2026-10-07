# SICAPv2 without the benign class

Splits: seeds 0, 1, 2, 3, 4. The benign class (NC) is removed from the training and test splits and the probe is fitted on the three Gleason patterns. Values are means over the splits, on the remaining test patches.

- Benign patches: 4417, from 10 of the 95 patients; patients with benign patches in a test split: 1.
- QWK: quadratic weighted kappa over the three Gleason patterns.
- E-AURC: area under the risk-coverage curve minus that of a perfect ranking at the same error rate; confidence is the maximum softmax probability.
- E-AURC, random: expected E-AURC of a ranking that is independent of correctness.
- ECE: top-label expected calibration error.

| Encoder | accuracy | QWK | E-AURC | E-AURC, random | ECE |
|---|---:|---:|---:|---:|---:|
| Lunit | 0.787 | 0.707 | 0.062 | 0.188 | 0.026 |
| CTransPath | 0.788 | 0.711 | 0.061 | 0.188 | 0.021 |
| CONCH | 0.795 | 0.726 | 0.060 | 0.182 | 0.030 |
| Hibou-B | 0.798 | 0.729 | 0.056 | 0.180 | 0.024 |
| Phikon | 0.738 | 0.622 | 0.092 | 0.223 | 0.051 |
| Phikon-v2 | 0.736 | 0.623 | 0.087 | 0.225 | 0.049 |
| Hibou-L | 0.825 | 0.773 | 0.044 | 0.158 | 0.023 |
| Virchow2 | 0.826 | 0.766 | 0.046 | 0.158 | 0.027 |
| UNI2 | 0.825 | 0.767 | 0.042 | 0.159 | 0.023 |
| H-optimus-0 | 0.810 | 0.742 | 0.050 | 0.170 | 0.047 |
| Midnight | 0.832 | 0.781 | 0.041 | 0.153 | 0.019 |
| DINOv2-nat | 0.744 | 0.613 | 0.094 | 0.219 | 0.032 |
| ImageNet-ViT | 0.729 | 0.593 | 0.106 | 0.230 | 0.038 |
