# Sensitivity of the detection AUROC to scorer settings

Stored split; scorers fitted on the training split; in-distribution set: the test split; the out-of-distribution set is the positive class.

- Tasks: NCT-CRC-HE, LC25000 lung, LC25000 colon, BreaKHis, PatchCamelyon, SICAPv2, PANDA.
- far: the out-of-distribution set is the test split of the donor task (7 conditions per encoder).
- blur: the out-of-distribution set is the Gaussian-blurred test split, sigma in {1.0, 2.0, 3.0}, categorical tasks (15 conditions per encoder).
- Each cell is the mean AUROC of one encoder over the conditions of the tier.
- normalise=True: embeddings are L2-normalised before the scorer; normalise=False: raw embeddings.
- k: the kNN score is the distance to the k-th nearest training embedding.
- covariance=ledoit_wolf: pooled within-class covariance with Ledoit-Wolf shrinkage; covariance=empirical: the same covariance without shrinkage (pseudo-inverse).

## Feature normalisation, Mahalanobis

| Encoder | far, normalise=False | far, normalise=True | blur, normalise=False | blur, normalise=True |
|---|---:|---:|---:|---:|
| Lunit | 1.000 | 0.999 | 0.970 | 0.968 |
| CTransPath | 0.998 | 0.998 | 0.904 | 0.896 |
| CONCH | 0.994 | 0.994 | 0.932 | 0.932 |
| Hibou-B | 0.999 | 0.998 | 0.901 | 0.898 |
| Phikon | 1.000 | 1.000 | 0.902 | 0.899 |
| Phikon-v2 | 1.000 | 1.000 | 0.908 | 0.911 |
| Hibou-L | 0.983 | 0.981 | 0.858 | 0.853 |
| Virchow2 | 0.998 | 0.992 | 0.902 | 0.887 |
| UNI2 | 0.999 | 1.000 | 0.834 | 0.829 |
| H-optimus-0 | 1.000 | 1.000 | 0.899 | 0.894 |
| Midnight | 0.999 | 0.999 | 0.949 | 0.944 |
| DINOv2-nat | 0.882 | 0.878 | 0.742 | 0.739 |
| ImageNet-ViT | 0.919 | 0.964 | 0.857 | 0.892 |

## Feature normalisation, kNN (k = 50)

| Encoder | far, normalise=False | far, normalise=True | blur, normalise=False | blur, normalise=True |
|---|---:|---:|---:|---:|
| Lunit | 0.999 | 0.999 | 0.786 | 0.784 |
| CTransPath | 0.996 | 0.998 | 0.719 | 0.726 |
| CONCH | 0.988 | 0.988 | 0.857 | 0.857 |
| Hibou-B | 0.994 | 0.994 | 0.809 | 0.805 |
| Phikon | 1.000 | 1.000 | 0.713 | 0.706 |
| Phikon-v2 | 1.000 | 1.000 | 0.787 | 0.788 |
| Hibou-L | 0.928 | 0.918 | 0.725 | 0.720 |
| Virchow2 | 0.992 | 0.957 | 0.734 | 0.713 |
| UNI2 | 0.983 | 1.000 | 0.650 | 0.667 |
| H-optimus-0 | 0.999 | 0.999 | 0.764 | 0.752 |
| Midnight | 0.992 | 0.992 | 0.771 | 0.763 |
| DINOv2-nat | 0.841 | 0.836 | 0.695 | 0.692 |
| ImageNet-ViT | 0.760 | 0.858 | 0.754 | 0.809 |

## Neighbourhood size, kNN on L2-normalised embeddings

| Encoder | far, k=1 | far, k=10 | far, k=50 | far, k=200 | blur, k=1 | blur, k=10 | blur, k=50 | blur, k=200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Lunit | 0.999 | 0.999 | 0.999 | 0.998 | 0.894 | 0.833 | 0.784 | 0.758 |
| CTransPath | 0.997 | 0.998 | 0.998 | 0.997 | 0.829 | 0.771 | 0.726 | 0.707 |
| CONCH | 0.987 | 0.988 | 0.988 | 0.985 | 0.909 | 0.891 | 0.857 | 0.827 |
| Hibou-B | 0.997 | 0.996 | 0.994 | 0.985 | 0.854 | 0.818 | 0.805 | 0.796 |
| Phikon | 0.994 | 1.000 | 1.000 | 0.999 | 0.813 | 0.749 | 0.706 | 0.681 |
| Phikon-v2 | 0.999 | 1.000 | 1.000 | 1.000 | 0.868 | 0.817 | 0.788 | 0.773 |
| Hibou-L | 0.951 | 0.937 | 0.918 | 0.892 | 0.799 | 0.747 | 0.720 | 0.706 |
| Virchow2 | 0.972 | 0.964 | 0.957 | 0.947 | 0.789 | 0.741 | 0.713 | 0.698 |
| UNI2 | 0.999 | 1.000 | 1.000 | 0.999 | 0.743 | 0.720 | 0.667 | 0.647 |
| H-optimus-0 | 0.995 | 0.999 | 0.999 | 0.998 | 0.824 | 0.795 | 0.752 | 0.732 |
| Midnight | 0.996 | 0.995 | 0.992 | 0.988 | 0.876 | 0.803 | 0.763 | 0.734 |
| DINOv2-nat | 0.863 | 0.851 | 0.836 | 0.821 | 0.736 | 0.711 | 0.692 | 0.682 |
| ImageNet-ViT | 0.897 | 0.879 | 0.858 | 0.831 | 0.858 | 0.833 | 0.809 | 0.790 |

## Covariance estimate, Mahalanobis on raw embeddings

| Encoder | far, covariance=ledoit_wolf | far, covariance=empirical | blur, covariance=ledoit_wolf | blur, covariance=empirical |
|---|---:|---:|---:|---:|
| Lunit | 1.000 | 1.000 | 0.970 | 0.974 |
| CTransPath | 0.998 | 0.998 | 0.904 | 0.908 |
| CONCH | 0.994 | 0.994 | 0.932 | 0.932 |
| Hibou-B | 0.999 | 0.999 | 0.901 | 0.919 |
| Phikon | 1.000 | 1.000 | 0.902 | 0.906 |
| Phikon-v2 | 1.000 | 1.000 | 0.908 | 0.910 |
| Hibou-L | 0.983 | 0.986 | 0.858 | 0.893 |
| Virchow2 | 0.998 | 0.997 | 0.902 | 0.915 |
| UNI2 | 0.999 | 0.999 | 0.834 | 0.854 |
| H-optimus-0 | 1.000 | 1.000 | 0.899 | 0.948 |
| Midnight | 0.999 | 0.999 | 0.949 | 0.957 |
| DINOv2-nat | 0.882 | 0.882 | 0.742 | 0.742 |
| ImageNet-ViT | 0.919 | 0.923 | 0.857 | 0.860 |
