# Linear-probe metrics

Each value is the mean over 5 train/test splits of the pooled embeddings (seeds 0, 1, 2, 3, 4; the test side is one fifth of the patches, or of the groups for tasks with patient or slide ids). Intervals are 95% percentile intervals of the bootstrap draws pooled over the splits (300 draws within each split). Tasks with patient or slide ids use a cluster bootstrap: groups are resampled with replacement and all their patches are taken. The other tasks resample patches.

## Accuracy on the categorical tasks

Columns: accuracy with its 95% interval per task; Mean: mean accuracy over NCT-CRC-HE, LC25000 lung, BreaKHis and PatchCamelyon.

| Encoder | NCT-CRC-HE | LC25000 lung | LC25000 colon | BreaKHis | PatchCamelyon | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Lunit | 0.988 [0.981, 0.994] | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] | 0.821 [0.564, 0.970] | 0.913 [0.857, 0.958] | 0.930 |
| CTransPath | 0.987 [0.977, 0.994] | 0.999 [0.998, 1.000] | 1.000 [1.000, 1.000] | 0.855 [0.633, 0.988] | 0.884 [0.794, 0.936] | 0.931 |
| CONCH | 0.981 [0.973, 0.988] | 0.999 [0.996, 1.000] | 1.000 [1.000, 1.000] | 0.799 [0.570, 0.962] | 0.910 [0.845, 0.955] | 0.922 |
| Hibou-B | 0.990 [0.984, 0.996] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.848 [0.634, 0.975] | 0.939 [0.891, 0.977] | 0.944 |
| Phikon | 0.994 [0.989, 0.999] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.869 [0.695, 0.992] | 0.931 [0.882, 0.971] | 0.948 |
| Phikon-v2 | 0.994 [0.987, 0.998] | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] | 0.879 [0.720, 0.990] | 0.927 [0.867, 0.968] | 0.950 |
| Hibou-L | 0.993 [0.986, 0.998] | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] | 0.869 [0.711, 0.985] | 0.950 [0.902, 0.984] | 0.953 |
| Virchow2 | 0.994 [0.988, 0.999] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.915 [0.784, 0.996] | 0.950 [0.911, 0.981] | 0.965 |
| UNI2 | 0.995 [0.988, 0.999] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.914 [0.772, 0.996] | 0.959 [0.926, 0.986] | 0.967 |
| H-optimus-0 | 0.995 [0.988, 0.999] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.879 [0.705, 0.992] | 0.949 [0.901, 0.982] | 0.956 |
| Midnight | 0.991 [0.986, 0.996] | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] | 0.900 [0.771, 0.993] | 0.941 [0.887, 0.978] | 0.958 |
| DINOv2-nat | 0.971 [0.961, 0.981] | 0.997 [0.994, 0.999] | 1.000 [0.999, 1.000] | 0.793 [0.571, 0.941] | 0.866 [0.781, 0.921] | 0.907 |
| ImageNet-ViT | 0.973 [0.961, 0.982] | 0.996 [0.994, 0.999] | 1.000 [1.000, 1.000] | 0.825 [0.577, 0.962] | 0.858 [0.763, 0.925] | 0.913 |

## Ordinal tasks

Columns, per task: Acc: accuracy; QWK: quadratic weighted kappa with 95% interval; AURC: area under the risk-coverage curve with 95% interval, uncertainty = 1 - maximum predicted probability; E-AURC: AURC minus the AURC of a perfect ranking at the same error rate; E-AURC random: expected E-AURC of a ranking independent of correctness; ECE: top-label expected calibration error (15 equal-width bins); C: median over splits of the selected inverse regularisation strength.

E-AURC random: with n test predictions of which m are errors, a ranking independent of correctness has expected selective risk m/n at every coverage, so its expected AURC is m/n. A perfect ranking has selective risk (k - (n - m)) / k after keeping the k most certain predictions for k > n - m and zero otherwise, so E-AURC random = m/n - (1/n) * sum_{k = n-m+1}^{n} (k - (n - m)) / k. For large n this tends to -(1 - e) * ln(1 - e) with e = m/n.

| Encoder | SICAPv2 Acc | SICAPv2 QWK | SICAPv2 AURC | SICAPv2 E-AURC | SICAPv2 E-AURC random | SICAPv2 ECE | SICAPv2 C | PANDA Acc | PANDA QWK | PANDA AURC | PANDA E-AURC | PANDA E-AURC random | PANDA ECE | PANDA C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lunit | 0.792 | 0.772 [0.400, 0.931] | 0.080 [0.026, 0.193] | 0.056 | 0.183 | 0.030 | 0.001 | 0.357 | 0.423 [0.317, 0.520] | 0.504 [0.403, 0.602] | 0.228 | 0.367 | 0.046 | 0.0001 |
| CTransPath | 0.796 | 0.789 [0.440, 0.913] | 0.077 [0.029, 0.186] | 0.054 | 0.181 | 0.024 | 0.001 | 0.363 | 0.455 [0.355, 0.543] | 0.504 [0.406, 0.606] | 0.234 | 0.367 | 0.021 | 0.0001 |
| CONCH | 0.813 | 0.828 [0.525, 0.929] | 0.068 [0.020, 0.195] | 0.049 | 0.167 | 0.034 | 0.001 | 0.345 | 0.437 [0.336, 0.529] | 0.529 [0.429, 0.637] | 0.240 | 0.366 | 0.032 | 0.0001 |
| Hibou-B | 0.807 | 0.800 [0.449, 0.925] | 0.067 [0.023, 0.162] | 0.046 | 0.172 | 0.025 | 0.001 | 0.362 | 0.445 [0.343, 0.536] | 0.504 [0.402, 0.608] | 0.234 | 0.367 | 0.026 | 0.0001 |
| Phikon | 0.749 | 0.735 [0.365, 0.874] | 0.110 [0.041, 0.247] | 0.074 | 0.216 | 0.036 | 0.001 | 0.356 | 0.444 [0.346, 0.530] | 0.517 [0.408, 0.622] | 0.240 | 0.367 | 0.026 | 0.0001 |
| Phikon-v2 | 0.747 | 0.734 [0.362, 0.883] | 0.108 [0.043, 0.240] | 0.072 | 0.216 | 0.034 | 0.001 | 0.356 | 0.433 [0.334, 0.518] | 0.519 [0.418, 0.620] | 0.242 | 0.367 | 0.023 | 0.0001 |
| Hibou-L | 0.843 | 0.846 [0.482, 0.956] | 0.049 [0.013, 0.136] | 0.035 | 0.143 | 0.026 | 0.001 | 0.373 | 0.469 [0.360, 0.558] | 0.486 [0.383, 0.592] | 0.227 | 0.367 | 0.027 | 0.0001 |
| Virchow2 | 0.852 | 0.873 [0.584, 0.964] | 0.043 [0.010, 0.125] | 0.031 | 0.136 | 0.018 | 0.001 | 0.390 | 0.511 [0.405, 0.601] | 0.465 [0.378, 0.564] | 0.222 | 0.367 | 0.026 | 0.0001 |
| UNI2 | 0.836 | 0.843 [0.519, 0.946] | 0.050 [0.017, 0.133] | 0.035 | 0.149 | 0.026 | 0.001 | 0.377 | 0.476 [0.378, 0.564] | 0.472 [0.380, 0.577] | 0.216 | 0.367 | 0.029 | 0.0001 |
| H-optimus-0 | 0.820 | 0.822 [0.518, 0.926] | 0.062 [0.023, 0.162] | 0.044 | 0.162 | 0.031 | 0.001 | 0.367 | 0.455 [0.355, 0.546] | 0.496 [0.406, 0.591] | 0.231 | 0.367 | 0.022 | 0.0001 |
| Midnight | 0.853 | 0.871 [0.585, 0.960] | 0.043 [0.011, 0.119] | 0.031 | 0.135 | 0.019 | 0.001 | 0.385 | 0.495 [0.381, 0.589] | 0.473 [0.366, 0.581] | 0.224 | 0.366 | 0.030 | 0.0001 |
| DINOv2-nat | 0.719 | 0.639 [0.244, 0.815] | 0.146 [0.061, 0.299] | 0.101 | 0.236 | 0.039 | 0.001 | 0.316 | 0.362 [0.263, 0.454] | 0.572 [0.471, 0.663] | 0.251 | 0.363 | 0.041 | 0.0001 |
| ImageNet-ViT | 0.713 | 0.634 [0.228, 0.827] | 0.150 [0.059, 0.303] | 0.103 | 0.240 | 0.036 | 0.001 | 0.314 | 0.348 [0.255, 0.439] | 0.577 [0.460, 0.678] | 0.254 | 0.363 | 0.032 | 0.0001 |

## PANDA by data provider

Columns, per provider: n: mean number of test patches; Acc: accuracy; QWK: quadratic weighted kappa with 95% interval, both computed on the test patches of that provider. QWK all: QWK on all test patches. QWK provider-only: QWK on all test patches of a baseline that predicts the most frequent training grade of the provider of each patch.

| Encoder | karolinska n | karolinska Acc | karolinska QWK | radboud n | radboud Acc | radboud QWK | QWK all | QWK provider-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Lunit | 3014 | 0.349 | 0.398 [0.261, 0.518] | 2344 | 0.367 | 0.455 [0.275, 0.618] | 0.423 [0.317, 0.520] | 0.000 |
| CTransPath | 3014 | 0.357 | 0.427 [0.303, 0.533] | 2344 | 0.370 | 0.493 [0.315, 0.648] | 0.455 [0.355, 0.543] | 0.000 |
| CONCH | 3014 | 0.360 | 0.429 [0.299, 0.537] | 2344 | 0.327 | 0.447 [0.261, 0.613] | 0.437 [0.336, 0.529] | 0.000 |
| Hibou-B | 3014 | 0.351 | 0.414 [0.262, 0.537] | 2344 | 0.377 | 0.487 [0.313, 0.637] | 0.445 [0.343, 0.536] | 0.000 |
| Phikon | 3014 | 0.359 | 0.434 [0.299, 0.550] | 2344 | 0.353 | 0.457 [0.281, 0.625] | 0.444 [0.346, 0.530] | 0.000 |
| Phikon-v2 | 3014 | 0.355 | 0.415 [0.285, 0.526] | 2344 | 0.355 | 0.456 [0.295, 0.603] | 0.433 [0.334, 0.518] | 0.000 |
| Hibou-L | 3014 | 0.368 | 0.439 [0.282, 0.559] | 2344 | 0.380 | 0.509 [0.324, 0.663] | 0.469 [0.360, 0.558] | 0.000 |
| Virchow2 | 3014 | 0.383 | 0.489 [0.353, 0.597] | 2344 | 0.400 | 0.540 [0.354, 0.690] | 0.511 [0.405, 0.601] | 0.000 |
| UNI2 | 3014 | 0.367 | 0.449 [0.318, 0.564] | 2344 | 0.390 | 0.512 [0.331, 0.667] | 0.476 [0.378, 0.564] | 0.000 |
| H-optimus-0 | 3014 | 0.364 | 0.440 [0.306, 0.558] | 2344 | 0.371 | 0.473 [0.300, 0.620] | 0.455 [0.355, 0.546] | 0.000 |
| Midnight | 3014 | 0.378 | 0.467 [0.336, 0.573] | 2344 | 0.394 | 0.533 [0.330, 0.684] | 0.495 [0.381, 0.589] | 0.000 |
| DINOv2-nat | 3014 | 0.306 | 0.338 [0.210, 0.450] | 2344 | 0.329 | 0.395 [0.233, 0.550] | 0.362 [0.263, 0.454] | 0.000 |
| ImageNet-ViT | 3014 | 0.314 | 0.328 [0.200, 0.433] | 2344 | 0.314 | 0.376 [0.199, 0.548] | 0.348 [0.255, 0.439] | 0.000 |

