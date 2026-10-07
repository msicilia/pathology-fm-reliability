# Out-of-distribution detection summary

AUROC of post-hoc novelty scorers fitted on the training split of each task, with the out-of-distribution set as the positive class. Tiers: Far: test split of another task; Near (categorical): one class held out, tasks NCT-CRC-HE, LC25000 lung; Near (grading): one grade held out, tasks SICAPv2, PANDA; Blur: the test split under Gaussian blur; Artefact: the classes BACK, DEB, ADI of NCT-CRC-HE held out jointly. n is the number of conditions (task and OOD set) per encoder.

## Mahalanobis AUROC per encoder and tier

Cells: mean +/- sample standard deviation (ddof = 1) over the n conditions of the tier (mean only when n = 1).

| Encoder | Far (n=7) | Near (categorical) (n=12) | Near (grading) (n=10) | Blur (n=15) | Artefact (n=1) | All (n=45) |
|---|---:|---:|---:|---:|---:|---:|
| Lunit | 1.000 +/- 0.001 | 0.970 +/- 0.030 | 0.545 +/- 0.148 | 0.970 +/- 0.054 | 0.993 | 0.881 +/- 0.197 |
| CTransPath | 0.998 +/- 0.003 | 0.988 +/- 0.014 | 0.544 +/- 0.129 | 0.904 +/- 0.125 | 0.996 | 0.863 +/- 0.200 |
| CONCH | 0.994 +/- 0.013 | 0.983 +/- 0.021 | 0.554 +/- 0.130 | 0.932 +/- 0.118 | 0.994 | 0.873 +/- 0.196 |
| Hibou-B | 0.999 +/- 0.004 | 0.982 +/- 0.021 | 0.563 +/- 0.150 | 0.901 +/- 0.155 | 0.993 | 0.864 +/- 0.201 |
| Phikon | 1.000 +/- 0.000 | 0.997 +/- 0.003 | 0.540 +/- 0.077 | 0.902 +/- 0.115 | 0.999 | 0.864 +/- 0.195 |
| Phikon-v2 | 1.000 +/- 0.000 | 0.991 +/- 0.012 | 0.542 +/- 0.067 | 0.908 +/- 0.126 | 0.992 | 0.865 +/- 0.195 |
| Hibou-L | 0.983 +/- 0.043 | 0.899 +/- 0.105 | 0.565 +/- 0.125 | 0.858 +/- 0.173 | 0.968 | 0.826 +/- 0.193 |
| Virchow2 | 0.998 +/- 0.005 | 0.996 +/- 0.006 | 0.578 +/- 0.142 | 0.902 +/- 0.151 | 0.996 | 0.872 +/- 0.196 |
| UNI2 | 0.999 +/- 0.001 | 0.983 +/- 0.019 | 0.568 +/- 0.174 | 0.834 +/- 0.164 | 0.997 | 0.844 +/- 0.205 |
| H-optimus-0 | 1.000 +/- 0.000 | 0.998 +/- 0.002 | 0.529 +/- 0.117 | 0.899 +/- 0.151 | 0.999 | 0.861 +/- 0.210 |
| Midnight | 0.999 +/- 0.002 | 0.997 +/- 0.003 | 0.551 +/- 0.168 | 0.949 +/- 0.089 | 0.997 | 0.882 +/- 0.202 |
| DINOv2-nat | 0.882 +/- 0.118 | 0.757 +/- 0.200 | 0.524 +/- 0.070 | 0.742 +/- 0.191 | 0.915 | 0.723 +/- 0.197 |
| ImageNet-ViT | 0.919 +/- 0.069 | 0.813 +/- 0.151 | 0.520 +/- 0.037 | 0.857 +/- 0.147 | 0.914 | 0.781 +/- 0.186 |

## kNN AUROC per encoder and tier

Cells: mean +/- sample standard deviation (ddof = 1) over the n conditions of the tier (mean only when n = 1).

| Encoder | Far (n=7) | Near (categorical) (n=12) | Near (grading) (n=10) | Blur (n=15) | Artefact (n=1) | All (n=45) |
|---|---:|---:|---:|---:|---:|---:|
| Lunit | 0.999 +/- 0.003 | 0.984 +/- 0.015 | 0.545 +/- 0.165 | 0.784 +/- 0.150 | 0.996 | 0.823 +/- 0.209 |
| CTransPath | 0.998 +/- 0.004 | 0.979 +/- 0.015 | 0.548 +/- 0.144 | 0.726 +/- 0.166 | 0.992 | 0.802 +/- 0.212 |
| CONCH | 0.988 +/- 0.023 | 0.981 +/- 0.015 | 0.548 +/- 0.141 | 0.857 +/- 0.150 | 0.992 | 0.845 +/- 0.201 |
| Hibou-B | 0.994 +/- 0.011 | 0.927 +/- 0.051 | 0.560 +/- 0.139 | 0.805 +/- 0.208 | 0.977 | 0.816 +/- 0.206 |
| Phikon | 1.000 +/- 0.000 | 0.980 +/- 0.015 | 0.542 +/- 0.083 | 0.706 +/- 0.157 | 0.983 | 0.794 +/- 0.208 |
| Phikon-v2 | 1.000 +/- 0.000 | 0.958 +/- 0.028 | 0.535 +/- 0.079 | 0.788 +/- 0.171 | 0.963 | 0.814 +/- 0.201 |
| Hibou-L | 0.918 +/- 0.135 | 0.728 +/- 0.178 | 0.554 +/- 0.103 | 0.720 +/- 0.205 | 0.833 | 0.719 +/- 0.197 |
| Virchow2 | 0.957 +/- 0.084 | 0.937 +/- 0.065 | 0.559 +/- 0.103 | 0.713 +/- 0.197 | 0.933 | 0.781 +/- 0.203 |
| UNI2 | 1.000 +/- 0.001 | 0.972 +/- 0.038 | 0.545 +/- 0.177 | 0.667 +/- 0.153 | 0.994 | 0.780 +/- 0.223 |
| H-optimus-0 | 0.999 +/- 0.002 | 0.961 +/- 0.073 | 0.523 +/- 0.127 | 0.752 +/- 0.187 | 0.997 | 0.801 +/- 0.220 |
| Midnight | 0.992 +/- 0.017 | 0.985 +/- 0.012 | 0.539 +/- 0.143 | 0.763 +/- 0.165 | 0.996 | 0.813 +/- 0.212 |
| DINOv2-nat | 0.836 +/- 0.142 | 0.717 +/- 0.201 | 0.519 +/- 0.074 | 0.692 +/- 0.182 | 0.920 | 0.688 +/- 0.190 |
| ImageNet-ViT | 0.858 +/- 0.105 | 0.776 +/- 0.156 | 0.519 +/- 0.040 | 0.809 +/- 0.158 | 0.917 | 0.746 +/- 0.179 |

## Scorer comparison

Cells: mean AUROC over all encoders and conditions of the tier; n is the number of (encoder, condition) pairs (13 encoders).

| Scorer | Far (n=91) | Near (categorical) (n=156) | Near (grading) (n=130) | Blur (n=195) | Artefact (n=13) | All (n=585) |
|---|---:|---:|---:|---:|---:|---:|
| Mahalanobis | 0.982 | 0.950 | 0.548 | 0.889 | 0.981 | 0.846 |
| kNN | 0.964 | 0.914 | 0.541 | 0.753 | 0.961 | 0.786 |
| Energy | 0.730 | 0.891 | 0.544 | 0.580 | 0.963 | 0.687 |
| MSP | 0.724 | 0.887 | 0.541 | 0.576 | 0.959 | 0.683 |

## Blur: Mahalanobis AUROC per encoder and sigma

Cells: AUROC of the blurred against the unblurred test split, mean over the 5 tasks (NCT-CRC-HE, LC25000 lung, LC25000 colon, BreaKHis, PatchCamelyon). Sigma is in pixels of a 224 x 224 input.

| Encoder | sigma = 1.0 | sigma = 2.0 | sigma = 3.0 |
|---|---:|---:|---:|
| Lunit | 0.919 | 0.992 | 0.998 |
| CTransPath | 0.754 | 0.961 | 0.997 |
| CONCH | 0.822 | 0.978 | 0.995 |
| Hibou-B | 0.723 | 0.979 | 1.000 |
| Phikon | 0.772 | 0.948 | 0.988 |
| Phikon-v2 | 0.794 | 0.931 | 0.998 |
| Hibou-L | 0.676 | 0.901 | 0.997 |
| Virchow2 | 0.764 | 0.943 | 1.000 |
| UNI2 | 0.664 | 0.869 | 0.970 |
| H-optimus-0 | 0.780 | 0.926 | 0.991 |
| Midnight | 0.869 | 0.980 | 1.000 |
| DINOv2-nat | 0.561 | 0.743 | 0.922 |
| ImageNet-ViT | 0.733 | 0.888 | 0.949 |

## Blur: AUROC per encoder and scorer

Cells: mean AUROC over the 3 values of sigma and the 5 tasks.

| Encoder | Mahalanobis | kNN | Energy | MSP |
|---|---:|---:|---:|---:|
| Lunit | 0.970 | 0.784 | 0.542 | 0.534 |
| CTransPath | 0.904 | 0.726 | 0.517 | 0.511 |
| CONCH | 0.932 | 0.857 | 0.680 | 0.669 |
| Hibou-B | 0.901 | 0.805 | 0.577 | 0.575 |
| Phikon | 0.902 | 0.706 | 0.576 | 0.575 |
| Phikon-v2 | 0.908 | 0.788 | 0.585 | 0.581 |
| Hibou-L | 0.858 | 0.720 | 0.508 | 0.518 |
| Virchow2 | 0.902 | 0.713 | 0.617 | 0.613 |
| UNI2 | 0.834 | 0.667 | 0.624 | 0.620 |
| H-optimus-0 | 0.899 | 0.752 | 0.636 | 0.631 |
| Midnight | 0.949 | 0.763 | 0.618 | 0.612 |
| DINOv2-nat | 0.742 | 0.692 | 0.472 | 0.484 |
| ImageNet-ViT | 0.857 | 0.809 | 0.581 | 0.571 |

## Near-OOD: Mahalanobis AUROC per encoder and task

Cells: mean AUROC over the held-out classes of the task (n classes).

| Encoder | NCT-CRC-HE (n=9) | LC25000 lung (n=3) | SICAPv2 (n=4) | PANDA (n=6) |
|---|---:|---:|---:|---:|
| Lunit | 0.960 | 0.999 | 0.590 | 0.515 |
| CTransPath | 0.984 | 0.999 | 0.591 | 0.512 |
| CONCH | 0.978 | 0.998 | 0.619 | 0.511 |
| Hibou-B | 0.976 | 1.000 | 0.633 | 0.516 |
| Phikon | 0.996 | 1.000 | 0.574 | 0.518 |
| Phikon-v2 | 0.988 | 0.999 | 0.578 | 0.518 |
| Hibou-L | 0.866 | 0.999 | 0.642 | 0.514 |
| Virchow2 | 0.994 | 1.000 | 0.667 | 0.519 |
| UNI2 | 0.977 | 1.000 | 0.643 | 0.517 |
| H-optimus-0 | 0.997 | 1.000 | 0.552 | 0.514 |
| Midnight | 0.996 | 1.000 | 0.611 | 0.510 |
| DINOv2-nat | 0.704 | 0.918 | 0.551 | 0.505 |
| ImageNet-ViT | 0.768 | 0.949 | 0.542 | 0.505 |

## Rank correlations across encoders

Spearman rho with its two-sided p-value. The 95% interval is a percentile bootstrap over encoders (2000 resamples with replacement, seed 0; resamples in which a variable is constant are left out). Encoders: all, or pathology (without DINOv2-nat and ImageNet-ViT).

### Probe accuracy and Mahalanobis AUROC

x: mean probe accuracy over NCT-CRC-HE, LC25000 lung, BreaKHis, PatchCamelyon (mean over the splits of probe_summary.csv); y: mean Mahalanobis AUROC of the encoder over the conditions of the tier.

| Tier | Encoders | n | rho | 95% interval | p |
|---|---:|---:|---:|---:|---:|
| Far | all | 13 | 0.467 | [-0.143, 0.894] | 0.108 |
| Near (categorical) | all | 13 | 0.643 | [0.127, 0.922] | 0.0178 |
| Near (grading) | all | 13 | 0.599 | [-0.020, 0.888] | 0.0306 |
| Blur | all | 13 | 0.000 | [-0.727, 0.654] | 1 |
| Artefact | all | 13 | 0.599 | [0.075, 0.849] | 0.0306 |
| All conditions | all | 13 | 0.220 | [-0.535, 0.783] | 0.471 |
| Far | pathology | 11 | 0.118 | [-0.500, 0.706] | 0.729 |
| Near (categorical) | pathology | 11 | 0.409 | [-0.171, 0.800] | 0.212 |
| Near (grading) | pathology | 11 | 0.345 | [-0.484, 0.792] | 0.298 |
| Blur | pathology | 11 | -0.555 | [-0.933, 0.138] | 0.0767 |
| Artefact | pathology | 11 | 0.345 | [-0.286, 0.701] | 0.298 |
| All conditions | pathology | 11 | -0.291 | [-0.840, 0.508] | 0.385 |

### Mahalanobis AUROC and kNN AUROC

x and y: mean AUROC of the encoder over the conditions of the tier under each scorer.

| Tier | Encoders | n | rho | 95% interval | p |
|---|---:|---:|---:|---:|---:|
| Far | all | 13 | 0.945 | [0.763, 0.994] | 1.12e-06 |
| Near (categorical) | all | 13 | 0.527 | [-0.167, 0.892] | 0.064 |
| Near (grading) | all | 13 | 0.835 | [0.462, 0.983] | 0.00038 |
| Blur | all | 13 | 0.473 | [-0.183, 0.862] | 0.103 |
| Artefact | all | 13 | 0.720 | [0.199, 0.954] | 0.00554 |
| All conditions | all | 13 | 0.819 | [0.429, 0.966] | 0.000621 |

## Values per encoder

Accuracy: mean probe accuracy over the tasks named above; Mahalanobis and kNN: mean AUROC over all conditions.

| Encoder | Accuracy | Mahalanobis | kNN |
|---|---:|---:|---:|
| Lunit | 0.930 | 0.881 | 0.823 |
| CTransPath | 0.931 | 0.863 | 0.802 |
| CONCH | 0.922 | 0.873 | 0.845 |
| Hibou-B | 0.944 | 0.864 | 0.816 |
| Phikon | 0.948 | 0.864 | 0.794 |
| Phikon-v2 | 0.950 | 0.865 | 0.814 |
| Hibou-L | 0.953 | 0.826 | 0.719 |
| Virchow2 | 0.965 | 0.872 | 0.781 |
| UNI2 | 0.967 | 0.844 | 0.780 |
| H-optimus-0 | 0.956 | 0.861 | 0.801 |
| Midnight | 0.958 | 0.882 | 0.813 |
| DINOv2-nat | 0.907 | 0.723 | 0.688 |
| ImageNet-ViT | 0.913 | 0.781 | 0.746 |

