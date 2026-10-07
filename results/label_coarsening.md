# Label coarsening on SICAPv2

Arms, on the same patient-grouped splits (seeds [0, 1, 2, 3, 4]):

- patch: probe trained and evaluated on the region-level patch labels.
- slide: every patch relabelled with the most severe grade among the patches of its slide (maximum over all exported patches of the slide); probe trained on the relabelled training patches.
- slide_abmil: gated attention aggregator (Ilse et al., 2018) trained on slide bags with the slide label and evaluated on the test slides; projection to 256 units, attention width 128, Adam, learning rate 0.0001, weight decay 0.0001, 30 epochs, one bag per step, embeddings standardised with the training patches, `torch.manual_seed` set to the split seed, weights of the final epoch. Device of this run: cpu. Results obtained on GPU backends may differ in the last digits.

## Relabelled patches

All exported patches (12081 patches, 155 slides). Rows: region-level label. Slide-label columns: number of patches of that row whose slide label is the column grade. Relabelled: patches whose slide label differs from their region-level label. Fraction: relabelled / patches.

| Region-level label | Patches | Slide label NC | Slide label G3 | Slide label G4 | Slide label G5 | Relabelled | Fraction |
|---|---|---|---|---|---|---|---|
| NC | 4417 | 4417 | 0 | 0 | 0 | 0 | 0.0000 |
| G3 | 2222 | 0 | 1015 | 1008 | 199 | 1207 | 0.5432 |
| G4 | 4494 | 0 | 0 | 3028 | 1466 | 1466 | 0.3262 |
| G5 | 948 | 0 | 0 | 0 | 948 | 0 | 0.0000 |
| all | 12081 | 4417 | 1015 | 4036 | 2613 | 2673 | 0.2213 |

## Metrics

Mean +/- sample standard deviation (ddof = 1) over the splits.

- Arm / evaluation: the arm and the test labels the predictions are compared with (patch_labels = region-level labels; slide_labels = labels after relabelling).
- Unit: what one prediction refers to (patch or slide). Test n: mean number of test units per split.
- Accuracy: fraction of correct predictions. QWK: quadratic weighted kappa.
- AURC: area under the risk-coverage curve, abstaining on the least confident predictions first (confidence = maximum softmax probability). E-AURC: AURC minus the AURC of a perfect ranking at the same error rate.
- ECE: top-label expected calibration error, 15 equal-width bins.
- Selected C: geometric mean over the splits of the inverse regularisation strength selected by cross-validation.

| Encoder | Arm / evaluation | Unit | Test n | Accuracy | QWK | AURC | E-AURC | ECE | Selected C |
|---|---|---|---|---|---|---|---|---|---|
| Lunit | patch / patch_labels | patch | 2047.0 | 0.792 +/- 0.043 | 0.772 +/- 0.113 | 0.080 +/- 0.029 | 0.056 +/- 0.019 | 0.030 +/- 0.016 | 0.001 |
| Lunit | slide / slide_labels | patch | 2047.0 | 0.616 +/- 0.073 | 0.684 +/- 0.145 | 0.237 +/- 0.087 | 0.148 +/- 0.054 | 0.110 +/- 0.059 | 0.000631 |
| Lunit | slide / patch_labels | patch | 2047.0 | 0.623 +/- 0.078 | 0.672 +/- 0.137 | 0.223 +/- 0.096 | 0.137 +/- 0.060 | 0.110 +/- 0.054 | 0.000631 |
| Lunit | slide_abmil / slide_labels | slide | 27.8 | 0.748 +/- 0.106 | 0.789 +/- 0.171 | 0.154 +/- 0.068 | 0.108 +/- 0.036 | 0.227 +/- 0.069 | n/a |
| CTransPath | patch / patch_labels | patch | 2047.0 | 0.796 +/- 0.034 | 0.789 +/- 0.095 | 0.077 +/- 0.025 | 0.054 +/- 0.017 | 0.024 +/- 0.010 | 0.001 |
| CTransPath | slide / slide_labels | patch | 2047.0 | 0.619 +/- 0.096 | 0.680 +/- 0.154 | 0.235 +/- 0.107 | 0.145 +/- 0.057 | 0.093 +/- 0.056 | 0.000251 |
| CTransPath | slide / patch_labels | patch | 2047.0 | 0.611 +/- 0.099 | 0.677 +/- 0.130 | 0.229 +/- 0.115 | 0.135 +/- 0.067 | 0.101 +/- 0.056 | 0.000251 |
| CTransPath | slide_abmil / slide_labels | slide | 27.8 | 0.711 +/- 0.076 | 0.767 +/- 0.058 | 0.169 +/- 0.037 | 0.114 +/- 0.027 | 0.243 +/- 0.066 | n/a |
| CONCH | patch / patch_labels | patch | 2047.0 | 0.813 +/- 0.037 | 0.828 +/- 0.076 | 0.068 +/- 0.030 | 0.049 +/- 0.022 | 0.034 +/- 0.006 | 0.00158 |
| CONCH | slide / slide_labels | patch | 2047.0 | 0.652 +/- 0.067 | 0.743 +/- 0.111 | 0.198 +/- 0.074 | 0.126 +/- 0.046 | 0.092 +/- 0.039 | 0.001 |
| CONCH | slide / patch_labels | patch | 2047.0 | 0.638 +/- 0.070 | 0.726 +/- 0.107 | 0.207 +/- 0.075 | 0.129 +/- 0.043 | 0.100 +/- 0.046 | 0.001 |
| CONCH | slide_abmil / slide_labels | slide | 27.8 | 0.744 +/- 0.082 | 0.816 +/- 0.097 | 0.101 +/- 0.031 | 0.057 +/- 0.015 | 0.218 +/- 0.042 | n/a |
| Hibou-B | patch / patch_labels | patch | 2047.0 | 0.807 +/- 0.036 | 0.800 +/- 0.102 | 0.067 +/- 0.023 | 0.046 +/- 0.014 | 0.025 +/- 0.012 | 0.001 |
| Hibou-B | slide / slide_labels | patch | 2047.0 | 0.636 +/- 0.087 | 0.704 +/- 0.151 | 0.218 +/- 0.085 | 0.137 +/- 0.045 | 0.105 +/- 0.038 | 0.000398 |
| Hibou-B | slide / patch_labels | patch | 2047.0 | 0.637 +/- 0.079 | 0.693 +/- 0.136 | 0.212 +/- 0.091 | 0.132 +/- 0.056 | 0.106 +/- 0.050 | 0.000398 |
| Hibou-B | slide_abmil / slide_labels | slide | 27.8 | 0.755 +/- 0.077 | 0.827 +/- 0.074 | 0.149 +/- 0.036 | 0.109 +/- 0.039 | 0.229 +/- 0.052 | n/a |
| Phikon | patch / patch_labels | patch | 2047.0 | 0.749 +/- 0.041 | 0.735 +/- 0.104 | 0.110 +/- 0.033 | 0.074 +/- 0.021 | 0.036 +/- 0.017 | 0.001 |
| Phikon | slide / slide_labels | patch | 2047.0 | 0.579 +/- 0.089 | 0.645 +/- 0.140 | 0.277 +/- 0.085 | 0.167 +/- 0.035 | 0.101 +/- 0.043 | 0.000251 |
| Phikon | slide / patch_labels | patch | 2047.0 | 0.582 +/- 0.091 | 0.632 +/- 0.126 | 0.273 +/- 0.103 | 0.164 +/- 0.054 | 0.097 +/- 0.049 | 0.000251 |
| Phikon | slide_abmil / slide_labels | slide | 27.8 | 0.624 +/- 0.109 | 0.714 +/- 0.127 | 0.218 +/- 0.070 | 0.122 +/- 0.024 | 0.331 +/- 0.048 | n/a |
| Phikon-v2 | patch / patch_labels | patch | 2047.0 | 0.747 +/- 0.044 | 0.734 +/- 0.108 | 0.108 +/- 0.034 | 0.072 +/- 0.020 | 0.034 +/- 0.021 | 0.001 |
| Phikon-v2 | slide / slide_labels | patch | 2047.0 | 0.588 +/- 0.075 | 0.633 +/- 0.146 | 0.258 +/- 0.076 | 0.154 +/- 0.037 | 0.071 +/- 0.024 | 0.000158 |
| Phikon-v2 | slide / patch_labels | patch | 2047.0 | 0.567 +/- 0.088 | 0.614 +/- 0.137 | 0.280 +/- 0.101 | 0.163 +/- 0.052 | 0.087 +/- 0.029 | 0.000158 |
| Phikon-v2 | slide_abmil / slide_labels | slide | 27.8 | 0.736 +/- 0.075 | 0.746 +/- 0.080 | 0.127 +/- 0.044 | 0.081 +/- 0.017 | 0.203 +/- 0.071 | n/a |
| Hibou-L | patch / patch_labels | patch | 2047.0 | 0.843 +/- 0.038 | 0.846 +/- 0.096 | 0.049 +/- 0.021 | 0.035 +/- 0.014 | 0.026 +/- 0.013 | 0.001 |
| Hibou-L | slide / slide_labels | patch | 2047.0 | 0.647 +/- 0.086 | 0.730 +/- 0.151 | 0.203 +/- 0.079 | 0.128 +/- 0.044 | 0.099 +/- 0.040 | 0.000251 |
| Hibou-L | slide / patch_labels | patch | 2047.0 | 0.642 +/- 0.083 | 0.720 +/- 0.135 | 0.196 +/- 0.086 | 0.118 +/- 0.053 | 0.101 +/- 0.045 | 0.000251 |
| Hibou-L | slide_abmil / slide_labels | slide | 27.8 | 0.753 +/- 0.101 | 0.835 +/- 0.075 | 0.132 +/- 0.069 | 0.089 +/- 0.048 | 0.232 +/- 0.068 | n/a |
| Virchow2 | patch / patch_labels | patch | 2047.0 | 0.852 +/- 0.032 | 0.873 +/- 0.067 | 0.043 +/- 0.017 | 0.031 +/- 0.012 | 0.018 +/- 0.006 | 0.001 |
| Virchow2 | slide / slide_labels | patch | 2047.0 | 0.658 +/- 0.061 | 0.769 +/- 0.110 | 0.182 +/- 0.073 | 0.113 +/- 0.048 | 0.092 +/- 0.026 | 0.0001 |
| Virchow2 | slide / patch_labels | patch | 2047.0 | 0.638 +/- 0.081 | 0.731 +/- 0.124 | 0.199 +/- 0.097 | 0.120 +/- 0.064 | 0.107 +/- 0.039 | 0.0001 |
| Virchow2 | slide_abmil / slide_labels | slide | 27.8 | 0.683 +/- 0.051 | 0.809 +/- 0.034 | 0.240 +/- 0.079 | 0.176 +/- 0.085 | 0.291 +/- 0.036 | n/a |
| UNI2 | patch / patch_labels | patch | 2047.0 | 0.836 +/- 0.038 | 0.843 +/- 0.090 | 0.050 +/- 0.021 | 0.035 +/- 0.014 | 0.026 +/- 0.007 | 0.001 |
| UNI2 | slide / slide_labels | patch | 2047.0 | 0.673 +/- 0.061 | 0.742 +/- 0.128 | 0.170 +/- 0.071 | 0.108 +/- 0.046 | 0.056 +/- 0.026 | 0.0001 |
| UNI2 | slide / patch_labels | patch | 2047.0 | 0.657 +/- 0.074 | 0.725 +/- 0.126 | 0.179 +/- 0.082 | 0.108 +/- 0.052 | 0.073 +/- 0.019 | 0.0001 |
| UNI2 | slide_abmil / slide_labels | slide | 27.8 | 0.807 +/- 0.071 | 0.882 +/- 0.055 | 0.103 +/- 0.043 | 0.077 +/- 0.026 | 0.185 +/- 0.065 | n/a |
| H-optimus-0 | patch / patch_labels | patch | 2047.0 | 0.820 +/- 0.039 | 0.822 +/- 0.079 | 0.062 +/- 0.024 | 0.044 +/- 0.016 | 0.031 +/- 0.014 | 0.001 |
| H-optimus-0 | slide / slide_labels | patch | 2047.0 | 0.614 +/- 0.084 | 0.685 +/- 0.131 | 0.244 +/- 0.074 | 0.153 +/- 0.036 | 0.097 +/- 0.062 | 0.000158 |
| H-optimus-0 | slide / patch_labels | patch | 2047.0 | 0.620 +/- 0.087 | 0.682 +/- 0.114 | 0.232 +/- 0.098 | 0.144 +/- 0.056 | 0.093 +/- 0.066 | 0.000158 |
| H-optimus-0 | slide_abmil / slide_labels | slide | 27.8 | 0.716 +/- 0.127 | 0.821 +/- 0.097 | 0.186 +/- 0.069 | 0.127 +/- 0.036 | 0.271 +/- 0.116 | n/a |
| Midnight | patch / patch_labels | patch | 2047.0 | 0.853 +/- 0.031 | 0.871 +/- 0.070 | 0.043 +/- 0.018 | 0.031 +/- 0.013 | 0.019 +/- 0.007 | 0.001 |
| Midnight | slide / slide_labels | patch | 2047.0 | 0.661 +/- 0.059 | 0.771 +/- 0.110 | 0.178 +/- 0.066 | 0.110 +/- 0.042 | 0.091 +/- 0.023 | 0.0001 |
| Midnight | slide / patch_labels | patch | 2047.0 | 0.641 +/- 0.086 | 0.730 +/- 0.127 | 0.194 +/- 0.088 | 0.116 +/- 0.053 | 0.108 +/- 0.046 | 0.0001 |
| Midnight | slide_abmil / slide_labels | slide | 27.8 | 0.773 +/- 0.079 | 0.866 +/- 0.053 | 0.141 +/- 0.084 | 0.106 +/- 0.063 | 0.224 +/- 0.055 | n/a |
| DINOv2-nat | patch / patch_labels | patch | 2047.0 | 0.719 +/- 0.044 | 0.639 +/- 0.136 | 0.146 +/- 0.044 | 0.101 +/- 0.029 | 0.039 +/- 0.015 | 0.001 |
| DINOv2-nat | slide / slide_labels | patch | 2047.0 | 0.566 +/- 0.064 | 0.580 +/- 0.147 | 0.301 +/- 0.073 | 0.186 +/- 0.036 | 0.099 +/- 0.046 | 0.001 |
| DINOv2-nat | slide / patch_labels | patch | 2047.0 | 0.567 +/- 0.069 | 0.568 +/- 0.142 | 0.296 +/- 0.089 | 0.181 +/- 0.052 | 0.094 +/- 0.055 | 0.001 |
| DINOv2-nat | slide_abmil / slide_labels | slide | 27.8 | 0.604 +/- 0.087 | 0.606 +/- 0.146 | 0.299 +/- 0.115 | 0.194 +/- 0.098 | 0.324 +/- 0.084 | n/a |
| ImageNet-ViT | patch / patch_labels | patch | 2047.0 | 0.713 +/- 0.047 | 0.634 +/- 0.152 | 0.150 +/- 0.047 | 0.103 +/- 0.030 | 0.036 +/- 0.016 | 0.001 |
| ImageNet-ViT | slide / slide_labels | patch | 2047.0 | 0.579 +/- 0.075 | 0.583 +/- 0.162 | 0.279 +/- 0.090 | 0.171 +/- 0.048 | 0.082 +/- 0.060 | 0.001 |
| ImageNet-ViT | slide / patch_labels | patch | 2047.0 | 0.568 +/- 0.076 | 0.574 +/- 0.153 | 0.293 +/- 0.106 | 0.178 +/- 0.067 | 0.092 +/- 0.059 | 0.001 |
| ImageNet-ViT | slide_abmil / slide_labels | slide | 27.8 | 0.611 +/- 0.129 | 0.638 +/- 0.216 | 0.257 +/- 0.113 | 0.151 +/- 0.082 | 0.300 +/- 0.094 | n/a |
