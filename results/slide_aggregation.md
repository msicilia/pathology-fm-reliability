# Slide-level aggregation on PANDA

Slide-grouped splits stratified by grade and data provider, seeds [0, 1, 2, 3, 4]; 576 training and 144 test slides in the first split. Every aggregator is trained on the training slides and evaluated on the test slides. Entries are the mean +/- the sample standard deviation (ddof = 1) over the splits.

Aggregators:

- patch_vote: patch-level probe, majority vote of its patch predictions per slide (ties to the lowest grade); confidence = fraction of patches voting for the winning grade.
- mean_pool, max_pool: per-slide mean or maximum of the patch embeddings, then the probe; confidence = maximum softmax probability.
- abmil: gated attention pooling (Ilse et al., 2018): linear projection to 256 units with ReLU, gated attention with 128 hidden units, linear classifier; confidence = maximum softmax probability.
- transformer: linear projection to 256 units, a class token, one transformer encoder layer (4 heads, feed-forward width 512, dropout 0.1), linear classifier on the class token; confidence = maximum softmax probability.

Training of abmil and transformer, identical for every encoder: embeddings standardised with the per-feature mean and standard deviation of the training patches; Adam, learning rate 0.0001, weight decay 0.0001, 30 epochs, one bag per step, cross-entropy over the six grades, `torch.manual_seed` set to the split seed; no validation set and no model selection, weights of the final epoch. Device of this run: cpu. Results obtained on GPU backends may differ in the last digits.

## QWK

- Patch-level probe: QWK of the patch-level probe over the test patches, each patch labelled with the grade of its slide. Its unit is the patch.
- Other columns: QWK of each aggregator over the test slides.

| Encoder | Patch-level probe | patch_vote | mean_pool | max_pool | abmil | transformer |
|---|---|---|---|---|---|---|
| Lunit | 0.423 +/- 0.021 | 0.537 +/- 0.038 | 0.599 +/- 0.031 | 0.624 +/- 0.073 | 0.695 +/- 0.053 | 0.670 +/- 0.080 |
| CTransPath | 0.455 +/- 0.021 | 0.530 +/- 0.036 | 0.636 +/- 0.051 | 0.636 +/- 0.060 | 0.680 +/- 0.061 | 0.676 +/- 0.068 |
| CONCH | 0.437 +/- 0.029 | 0.531 +/- 0.061 | 0.644 +/- 0.065 | 0.655 +/- 0.069 | 0.715 +/- 0.076 | 0.728 +/- 0.065 |
| Hibou-B | 0.445 +/- 0.016 | 0.540 +/- 0.047 | 0.611 +/- 0.053 | 0.602 +/- 0.049 | 0.715 +/- 0.055 | 0.705 +/- 0.075 |
| Phikon | 0.444 +/- 0.015 | 0.530 +/- 0.023 | 0.562 +/- 0.055 | 0.526 +/- 0.034 | 0.624 +/- 0.077 | 0.608 +/- 0.035 |
| Phikon-v2 | 0.433 +/- 0.012 | 0.529 +/- 0.032 | 0.586 +/- 0.041 | 0.560 +/- 0.024 | 0.636 +/- 0.075 | 0.590 +/- 0.069 |
| Hibou-L | 0.469 +/- 0.015 | 0.585 +/- 0.023 | 0.658 +/- 0.056 | 0.632 +/- 0.050 | 0.722 +/- 0.074 | 0.740 +/- 0.032 |
| Virchow2 | 0.511 +/- 0.018 | 0.604 +/- 0.029 | 0.672 +/- 0.043 | 0.699 +/- 0.053 | 0.744 +/- 0.071 | 0.718 +/- 0.061 |
| UNI2 | 0.476 +/- 0.013 | 0.569 +/- 0.057 | 0.650 +/- 0.085 | 0.660 +/- 0.060 | 0.742 +/- 0.080 | 0.676 +/- 0.037 |
| H-optimus-0 | 0.455 +/- 0.023 | 0.549 +/- 0.037 | 0.611 +/- 0.029 | 0.600 +/- 0.069 | 0.679 +/- 0.034 | 0.676 +/- 0.057 |
| Midnight | 0.495 +/- 0.028 | 0.596 +/- 0.077 | 0.666 +/- 0.067 | 0.711 +/- 0.079 | 0.744 +/- 0.050 | 0.702 +/- 0.070 |
| DINOv2-nat | 0.362 +/- 0.033 | 0.532 +/- 0.059 | 0.520 +/- 0.024 | 0.542 +/- 0.051 | 0.522 +/- 0.048 | 0.541 +/- 0.050 |
| ImageNet-ViT | 0.348 +/- 0.027 | 0.486 +/- 0.060 | 0.515 +/- 0.077 | 0.493 +/- 0.054 | 0.528 +/- 0.079 | 0.573 +/- 0.071 |

## Slide-level accuracy, selective prediction and calibration

- Accuracy: fraction of test slides graded correctly.
- AURC: area under the risk-coverage curve, abstaining on the least confident slides first.
- E-AURC: AURC minus the AURC of a perfect ranking at the same error rate.
- ECE: top-label expected calibration error, 15 equal-width bins.
- Test slides: number of test slides per split (mean over the splits).

| Encoder | Aggregator | Accuracy | AURC | E-AURC | ECE | Test slides |
|---|---|---|---|---|---|---|
| Lunit | patch_vote | 0.401 +/- 0.025 | 0.453 +/- 0.054 | 0.219 +/- 0.036 | 0.183 +/- 0.024 | 144.0 |
| Lunit | mean_pool | 0.460 +/- 0.022 | 0.384 +/- 0.036 | 0.198 +/- 0.029 | 0.086 +/- 0.017 | 144.0 |
| Lunit | max_pool | 0.471 +/- 0.033 | 0.387 +/- 0.050 | 0.210 +/- 0.040 | 0.111 +/- 0.029 | 144.0 |
| Lunit | abmil | 0.506 +/- 0.033 | 0.340 +/- 0.020 | 0.188 +/- 0.011 | 0.349 +/- 0.029 | 144.0 |
| Lunit | transformer | 0.510 +/- 0.051 | 0.358 +/- 0.028 | 0.208 +/- 0.027 | 0.374 +/- 0.050 | 144.0 |
| CTransPath | patch_vote | 0.401 +/- 0.026 | 0.437 +/- 0.060 | 0.202 +/- 0.038 | 0.179 +/- 0.029 | 144.0 |
| CTransPath | mean_pool | 0.478 +/- 0.032 | 0.378 +/- 0.061 | 0.206 +/- 0.040 | 0.119 +/- 0.025 | 144.0 |
| CTransPath | max_pool | 0.464 +/- 0.043 | 0.378 +/- 0.059 | 0.195 +/- 0.033 | 0.113 +/- 0.038 | 144.0 |
| CTransPath | abmil | 0.496 +/- 0.053 | 0.369 +/- 0.033 | 0.208 +/- 0.040 | 0.381 +/- 0.042 | 144.0 |
| CTransPath | transformer | 0.507 +/- 0.029 | 0.371 +/- 0.041 | 0.220 +/- 0.032 | 0.381 +/- 0.029 | 144.0 |
| CONCH | patch_vote | 0.404 +/- 0.048 | 0.480 +/- 0.085 | 0.245 +/- 0.046 | 0.160 +/- 0.033 | 144.0 |
| CONCH | mean_pool | 0.447 +/- 0.035 | 0.410 +/- 0.040 | 0.214 +/- 0.030 | 0.114 +/- 0.014 | 144.0 |
| CONCH | max_pool | 0.439 +/- 0.021 | 0.422 +/- 0.048 | 0.220 +/- 0.036 | 0.097 +/- 0.016 | 144.0 |
| CONCH | abmil | 0.510 +/- 0.049 | 0.364 +/- 0.077 | 0.214 +/- 0.046 | 0.330 +/- 0.035 | 144.0 |
| CONCH | transformer | 0.507 +/- 0.045 | 0.354 +/- 0.035 | 0.202 +/- 0.019 | 0.356 +/- 0.041 | 144.0 |
| Hibou-B | patch_vote | 0.428 +/- 0.049 | 0.427 +/- 0.064 | 0.214 +/- 0.028 | 0.164 +/- 0.046 | 144.0 |
| Hibou-B | mean_pool | 0.496 +/- 0.021 | 0.367 +/- 0.033 | 0.208 +/- 0.026 | 0.121 +/- 0.034 | 144.0 |
| Hibou-B | max_pool | 0.469 +/- 0.044 | 0.362 +/- 0.034 | 0.183 +/- 0.014 | 0.112 +/- 0.021 | 144.0 |
| Hibou-B | abmil | 0.544 +/- 0.031 | 0.303 +/- 0.049 | 0.176 +/- 0.033 | 0.340 +/- 0.027 | 144.0 |
| Hibou-B | transformer | 0.535 +/- 0.032 | 0.351 +/- 0.036 | 0.218 +/- 0.036 | 0.363 +/- 0.032 | 144.0 |
| Phikon | patch_vote | 0.400 +/- 0.039 | 0.482 +/- 0.077 | 0.245 +/- 0.049 | 0.201 +/- 0.045 | 144.0 |
| Phikon | mean_pool | 0.451 +/- 0.046 | 0.403 +/- 0.065 | 0.210 +/- 0.043 | 0.107 +/- 0.035 | 144.0 |
| Phikon | max_pool | 0.422 +/- 0.053 | 0.444 +/- 0.053 | 0.226 +/- 0.036 | 0.111 +/- 0.038 | 144.0 |
| Phikon | abmil | 0.482 +/- 0.034 | 0.382 +/- 0.054 | 0.213 +/- 0.033 | 0.382 +/- 0.034 | 144.0 |
| Phikon | transformer | 0.485 +/- 0.037 | 0.399 +/- 0.047 | 0.232 +/- 0.040 | 0.416 +/- 0.029 | 144.0 |
| Phikon-v2 | patch_vote | 0.412 +/- 0.037 | 0.461 +/- 0.073 | 0.235 +/- 0.046 | 0.184 +/- 0.033 | 144.0 |
| Phikon-v2 | mean_pool | 0.458 +/- 0.027 | 0.410 +/- 0.057 | 0.224 +/- 0.045 | 0.117 +/- 0.029 | 144.0 |
| Phikon-v2 | max_pool | 0.421 +/- 0.025 | 0.461 +/- 0.073 | 0.244 +/- 0.057 | 0.126 +/- 0.013 | 144.0 |
| Phikon-v2 | abmil | 0.476 +/- 0.054 | 0.379 +/- 0.037 | 0.204 +/- 0.024 | 0.367 +/- 0.044 | 144.0 |
| Phikon-v2 | transformer | 0.469 +/- 0.066 | 0.382 +/- 0.074 | 0.202 +/- 0.032 | 0.415 +/- 0.064 | 144.0 |
| Hibou-L | patch_vote | 0.436 +/- 0.040 | 0.442 +/- 0.071 | 0.236 +/- 0.046 | 0.185 +/- 0.031 | 144.0 |
| Hibou-L | mean_pool | 0.460 +/- 0.026 | 0.373 +/- 0.033 | 0.188 +/- 0.017 | 0.116 +/- 0.020 | 144.0 |
| Hibou-L | max_pool | 0.442 +/- 0.026 | 0.391 +/- 0.036 | 0.191 +/- 0.029 | 0.101 +/- 0.028 | 144.0 |
| Hibou-L | abmil | 0.537 +/- 0.047 | 0.300 +/- 0.048 | 0.168 +/- 0.020 | 0.336 +/- 0.041 | 144.0 |
| Hibou-L | transformer | 0.550 +/- 0.017 | 0.313 +/- 0.041 | 0.190 +/- 0.034 | 0.336 +/- 0.017 | 144.0 |
| Virchow2 | patch_vote | 0.464 +/- 0.011 | 0.399 +/- 0.049 | 0.217 +/- 0.044 | 0.172 +/- 0.020 | 144.0 |
| Virchow2 | mean_pool | 0.499 +/- 0.021 | 0.336 +/- 0.033 | 0.180 +/- 0.023 | 0.092 +/- 0.019 | 144.0 |
| Virchow2 | max_pool | 0.485 +/- 0.032 | 0.349 +/- 0.047 | 0.182 +/- 0.028 | 0.110 +/- 0.044 | 144.0 |
| Virchow2 | abmil | 0.579 +/- 0.044 | 0.268 +/- 0.043 | 0.160 +/- 0.028 | 0.318 +/- 0.047 | 144.0 |
| Virchow2 | transformer | 0.568 +/- 0.028 | 0.292 +/- 0.036 | 0.179 +/- 0.022 | 0.322 +/- 0.037 | 144.0 |
| UNI2 | patch_vote | 0.456 +/- 0.038 | 0.421 +/- 0.065 | 0.232 +/- 0.040 | 0.150 +/- 0.024 | 144.0 |
| UNI2 | mean_pool | 0.506 +/- 0.033 | 0.343 +/- 0.050 | 0.191 +/- 0.036 | 0.124 +/- 0.017 | 144.0 |
| UNI2 | max_pool | 0.485 +/- 0.039 | 0.381 +/- 0.054 | 0.214 +/- 0.029 | 0.121 +/- 0.039 | 144.0 |
| UNI2 | abmil | 0.554 +/- 0.055 | 0.289 +/- 0.044 | 0.166 +/- 0.026 | 0.337 +/- 0.044 | 144.0 |
| UNI2 | transformer | 0.536 +/- 0.031 | 0.334 +/- 0.035 | 0.202 +/- 0.027 | 0.377 +/- 0.029 | 144.0 |
| H-optimus-0 | patch_vote | 0.443 +/- 0.035 | 0.427 +/- 0.054 | 0.228 +/- 0.028 | 0.142 +/- 0.032 | 144.0 |
| H-optimus-0 | mean_pool | 0.467 +/- 0.033 | 0.376 +/- 0.049 | 0.195 +/- 0.044 | 0.113 +/- 0.040 | 144.0 |
| H-optimus-0 | max_pool | 0.449 +/- 0.032 | 0.412 +/- 0.067 | 0.217 +/- 0.060 | 0.140 +/- 0.030 | 144.0 |
| H-optimus-0 | abmil | 0.497 +/- 0.038 | 0.372 +/- 0.023 | 0.213 +/- 0.020 | 0.371 +/- 0.034 | 144.0 |
| H-optimus-0 | transformer | 0.482 +/- 0.047 | 0.371 +/- 0.055 | 0.201 +/- 0.033 | 0.414 +/- 0.046 | 144.0 |
| Midnight | patch_vote | 0.453 +/- 0.047 | 0.421 +/- 0.058 | 0.228 +/- 0.025 | 0.176 +/- 0.031 | 144.0 |
| Midnight | mean_pool | 0.507 +/- 0.030 | 0.346 +/- 0.035 | 0.195 +/- 0.017 | 0.103 +/- 0.019 | 144.0 |
| Midnight | max_pool | 0.507 +/- 0.039 | 0.337 +/- 0.037 | 0.186 +/- 0.027 | 0.129 +/- 0.028 | 144.0 |
| Midnight | abmil | 0.547 +/- 0.023 | 0.321 +/- 0.017 | 0.196 +/- 0.022 | 0.361 +/- 0.017 | 144.0 |
| Midnight | transformer | 0.524 +/- 0.043 | 0.355 +/- 0.058 | 0.214 +/- 0.031 | 0.390 +/- 0.045 | 144.0 |
| DINOv2-nat | patch_vote | 0.396 +/- 0.043 | 0.452 +/- 0.062 | 0.211 +/- 0.034 | 0.134 +/- 0.041 | 144.0 |
| DINOv2-nat | mean_pool | 0.403 +/- 0.018 | 0.439 +/- 0.043 | 0.206 +/- 0.030 | 0.088 +/- 0.014 | 144.0 |
| DINOv2-nat | max_pool | 0.406 +/- 0.033 | 0.464 +/- 0.068 | 0.233 +/- 0.040 | 0.129 +/- 0.024 | 144.0 |
| DINOv2-nat | abmil | 0.414 +/- 0.032 | 0.496 +/- 0.056 | 0.272 +/- 0.027 | 0.430 +/- 0.037 | 144.0 |
| DINOv2-nat | transformer | 0.412 +/- 0.038 | 0.475 +/- 0.030 | 0.249 +/- 0.035 | 0.463 +/- 0.034 | 144.0 |
| ImageNet-ViT | patch_vote | 0.368 +/- 0.048 | 0.483 +/- 0.058 | 0.214 +/- 0.035 | 0.162 +/- 0.032 | 144.0 |
| ImageNet-ViT | mean_pool | 0.401 +/- 0.035 | 0.457 +/- 0.036 | 0.221 +/- 0.028 | 0.097 +/- 0.022 | 144.0 |
| ImageNet-ViT | max_pool | 0.376 +/- 0.034 | 0.515 +/- 0.040 | 0.256 +/- 0.024 | 0.093 +/- 0.029 | 144.0 |
| ImageNet-ViT | abmil | 0.383 +/- 0.048 | 0.505 +/- 0.048 | 0.252 +/- 0.026 | 0.459 +/- 0.047 | 144.0 |
| ImageNet-ViT | transformer | 0.382 +/- 0.021 | 0.514 +/- 0.030 | 0.261 +/- 0.033 | 0.503 +/- 0.024 | 144.0 |

## Patch-level probe

Accuracy and QWK of the patch-level probe over the test patches (unit: patch). AURC and ECE are not reported at this level.

| Encoder | Patch-level accuracy | Patch-level QWK | Test patches |
|---|---|---|---|
| Lunit | 0.357 +/- 0.024 | 0.423 +/- 0.021 | 5358.0 |
| CTransPath | 0.363 +/- 0.020 | 0.455 +/- 0.021 | 5358.0 |
| CONCH | 0.345 +/- 0.027 | 0.437 +/- 0.029 | 5358.0 |
| Hibou-B | 0.362 +/- 0.024 | 0.445 +/- 0.016 | 5358.0 |
| Phikon | 0.356 +/- 0.024 | 0.444 +/- 0.015 | 5358.0 |
| Phikon-v2 | 0.356 +/- 0.018 | 0.433 +/- 0.012 | 5358.0 |
| Hibou-L | 0.373 +/- 0.026 | 0.469 +/- 0.015 | 5358.0 |
| Virchow2 | 0.390 +/- 0.013 | 0.511 +/- 0.018 | 5358.0 |
| UNI2 | 0.377 +/- 0.024 | 0.476 +/- 0.013 | 5358.0 |
| H-optimus-0 | 0.367 +/- 0.019 | 0.455 +/- 0.023 | 5358.0 |
| Midnight | 0.385 +/- 0.030 | 0.495 +/- 0.028 | 5358.0 |
| DINOv2-nat | 0.316 +/- 0.023 | 0.362 +/- 0.033 | 5358.0 |
| ImageNet-ViT | 0.314 +/- 0.026 | 0.348 +/- 0.027 | 5358.0 |
