# Accuracy of the linear probe under Gaussian blur

Stored split. The probe is fitted on the training split and evaluated on the test split and on the same test patches under Gaussian blur. Sigma is in pixels of a 224 x 224 input.

- Accuracy: fraction of test patches classified correctly.
- Change: accuracy under blur minus accuracy of the unblurred test split.

## Mean over the 5 tasks (NCT-CRC-HE, LC25000 lung, LC25000 colon, BreaKHis, PatchCamelyon)

| Encoder | accuracy, unblurred | change, sigma=1 | change, sigma=2 | change, sigma=3 |
|---|---:|---:|---:|---:|
| Lunit | 0.956 | 0.001 | -0.010 | -0.030 |
| CTransPath | 0.956 | 0.004 | -0.001 | -0.033 |
| CONCH | 0.936 | -0.010 | -0.076 | -0.228 |
| Hibou-B | 0.961 | 0.002 | -0.014 | -0.082 |
| Phikon | 0.959 | 0.001 | 0.001 | -0.007 |
| Phikon-v2 | 0.965 | 0.000 | -0.009 | -0.029 |
| Hibou-L | 0.965 | 0.003 | -0.001 | -0.047 |
| Virchow2 | 0.979 | 0.000 | -0.001 | -0.134 |
| UNI2 | 0.979 | 0.001 | 0.000 | -0.006 |
| H-optimus-0 | 0.972 | 0.003 | 0.000 | -0.017 |
| Midnight | 0.974 | 0.000 | -0.002 | -0.010 |
| DINOv2-nat | 0.934 | 0.000 | -0.018 | -0.070 |
| ImageNet-ViT | 0.929 | -0.022 | -0.063 | -0.124 |

## NCT-CRC-HE

| Encoder | accuracy, unblurred | change, sigma=1 | change, sigma=2 | change, sigma=3 |
|---|---:|---:|---:|---:|
| Lunit | 0.990 | -0.001 | -0.011 | -0.034 |
| CTransPath | 0.991 | -0.002 | -0.007 | -0.051 |
| CONCH | 0.984 | -0.009 | -0.136 | -0.419 |
| Hibou-B | 0.991 | -0.001 | -0.002 | -0.152 |
| Phikon | 0.992 | 0.000 | -0.001 | -0.013 |
| Phikon-v2 | 0.992 | 0.001 | -0.001 | -0.015 |
| Hibou-L | 0.991 | 0.000 | -0.002 | -0.059 |
| Virchow2 | 0.993 | 0.001 | 0.000 | -0.034 |
| UNI2 | 0.994 | -0.001 | -0.002 | -0.013 |
| H-optimus-0 | 0.995 | 0.000 | -0.001 | -0.017 |
| Midnight | 0.989 | 0.001 | -0.001 | -0.012 |
| DINOv2-nat | 0.970 | 0.001 | -0.017 | -0.123 |
| ImageNet-ViT | 0.969 | -0.004 | -0.079 | -0.186 |

## LC25000 lung

| Encoder | accuracy, unblurred | change, sigma=1 | change, sigma=2 | change, sigma=3 |
|---|---:|---:|---:|---:|
| Lunit | 1.000 | -0.004 | -0.020 | -0.027 |
| CTransPath | 0.999 | 0.000 | -0.002 | -0.019 |
| CONCH | 0.999 | -0.015 | -0.051 | -0.288 |
| Hibou-B | 1.000 | 0.000 | 0.000 | -0.015 |
| Phikon | 1.000 | 0.000 | 0.000 | -0.022 |
| Phikon-v2 | 1.000 | 0.000 | -0.001 | -0.014 |
| Hibou-L | 1.000 | 0.000 | -0.001 | -0.006 |
| Virchow2 | 1.000 | 0.000 | 0.000 | -0.032 |
| UNI2 | 1.000 | 0.000 | 0.000 | -0.002 |
| H-optimus-0 | 1.000 | 0.000 | 0.000 | -0.002 |
| Midnight | 1.000 | 0.000 | -0.001 | -0.011 |
| DINOv2-nat | 0.999 | -0.002 | -0.026 | -0.045 |
| ImageNet-ViT | 0.996 | -0.017 | -0.054 | -0.088 |

## LC25000 colon

| Encoder | accuracy, unblurred | change, sigma=1 | change, sigma=2 | change, sigma=3 |
|---|---:|---:|---:|---:|
| Lunit | 1.000 | 0.000 | 0.000 | -0.002 |
| CTransPath | 1.000 | 0.000 | 0.000 | 0.000 |
| CONCH | 1.000 | -0.001 | -0.002 | -0.052 |
| Hibou-B | 1.000 | 0.000 | 0.000 | 0.000 |
| Phikon | 1.000 | 0.000 | 0.000 | -0.001 |
| Phikon-v2 | 1.000 | 0.000 | 0.000 | 0.000 |
| Hibou-L | 1.000 | 0.000 | 0.000 | 0.000 |
| Virchow2 | 1.000 | 0.000 | 0.000 | -0.248 |
| UNI2 | 1.000 | 0.000 | 0.000 | -0.002 |
| H-optimus-0 | 1.000 | 0.000 | 0.000 | 0.000 |
| Midnight | 1.000 | 0.000 | -0.001 | 0.000 |
| DINOv2-nat | 1.000 | 0.000 | 0.000 | -0.002 |
| ImageNet-ViT | 1.000 | -0.001 | -0.001 | -0.036 |

## BreaKHis

| Encoder | accuracy, unblurred | change, sigma=1 | change, sigma=2 | change, sigma=3 |
|---|---:|---:|---:|---:|
| Lunit | 0.884 | -0.004 | -0.018 | -0.040 |
| CTransPath | 0.898 | -0.004 | -0.010 | -0.063 |
| CONCH | 0.788 | -0.002 | -0.059 | -0.151 |
| Hibou-B | 0.890 | 0.001 | -0.068 | -0.208 |
| Phikon | 0.884 | -0.001 | 0.000 | -0.012 |
| Phikon-v2 | 0.906 | -0.001 | -0.017 | -0.033 |
| Hibou-L | 0.898 | 0.005 | -0.014 | -0.128 |
| Virchow2 | 0.961 | -0.002 | -0.009 | -0.248 |
| UNI2 | 0.946 | 0.004 | -0.001 | -0.009 |
| H-optimus-0 | 0.923 | -0.003 | -0.010 | -0.039 |
| Midnight | 0.941 | -0.006 | -0.018 | -0.010 |
| DINOv2-nat | 0.824 | 0.009 | -0.039 | -0.101 |
| ImageNet-ViT | 0.798 | -0.064 | -0.135 | -0.167 |

## PatchCamelyon

| Encoder | accuracy, unblurred | change, sigma=1 | change, sigma=2 | change, sigma=3 |
|---|---:|---:|---:|---:|
| Lunit | 0.904 | 0.013 | -0.001 | -0.046 |
| CTransPath | 0.893 | 0.025 | 0.016 | -0.031 |
| CONCH | 0.907 | -0.022 | -0.133 | -0.227 |
| Hibou-B | 0.924 | 0.008 | 0.003 | -0.037 |
| Phikon | 0.919 | 0.006 | 0.008 | 0.015 |
| Phikon-v2 | 0.928 | 0.001 | -0.027 | -0.083 |
| Hibou-L | 0.935 | 0.010 | 0.014 | -0.041 |
| Virchow2 | 0.939 | -0.001 | 0.001 | -0.109 |
| UNI2 | 0.953 | -0.001 | 0.004 | -0.002 |
| H-optimus-0 | 0.944 | 0.017 | 0.010 | -0.025 |
| Midnight | 0.938 | 0.008 | 0.010 | -0.016 |
| DINOv2-nat | 0.879 | -0.006 | -0.008 | -0.080 |
| ImageNet-ViT | 0.881 | -0.024 | -0.047 | -0.145 |
