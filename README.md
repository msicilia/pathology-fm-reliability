# A Retraining-Free Reliability Audit and Abstention Gate for Frozen Pathology Foundation Models

This repository embeds the patches of six public histopathology datasets with 13 frozen
encoders and caches the embeddings. From the cached embeddings and a linear probe it computes
out-of-distribution detection, calibration and selective-prediction metrics for every encoder.
It also evaluates a two-threshold abstention rule that combines a novelty score with the
confidence of the probe.

## Requirements and installation

- [uv](https://docs.astral.sh/uv/) manages the environment. Python 3.11 or later is required;
  the code was run with Python 3.13.12.
- Hardware: the embedding export needs a CUDA GPU or Apple MPS. Every other stage runs on CPU.
- A Hugging Face account with access granted on the pages of the gated encoders (see the
  table below), and Kaggle API credentials for the PANDA download.

```sh
uv sync                 # analysis and embedding export
uv sync --extra data    # adds the dataset download and preparation dependencies
```

All commands are run from the repository root.

## Encoders

No encoder is trained or fine-tuned. Every input is resized on the shorter side with bicubic
interpolation, centre-cropped to the input size and normalised with the mean and standard
deviation listed below.

| Key | Display name | Source | Access | Input | Normalisation | Pooling | Dim. |
|---|---|---|---|---|---|---|---|
| `lunit` | Lunit | Hugging Face `1aurent/vit_small_patch8_224.lunit_dino` (timm) | open | 224 | mean 0.7032, 0.5361, 0.6610; std 0.2172, 0.2608, 0.2072 | class token | 384 |
| `ctranspath` | CTransPath | checkpoint from https://github.com/Xiyue-Wang/TransPath | see source | 224 | ImageNet | mean over the final feature map | 768 |
| `conch` | CONCH | Hugging Face `MahmoodLab/conch` | gated | 448 | CLIP | attentional pooler, no projection | 512 |
| `hibou_b` | Hibou-B | Hugging Face `histai/hibou-b` | gated | 224 | mean 0.7068, 0.5755, 0.7220; std 0.1950, 0.2316, 0.1816 | pooler output | 768 |
| `phikon` | Phikon | Hugging Face `owkin/phikon` | open | 224 | ImageNet | class token | 768 |
| `phikon_v2` | Phikon-v2 | Hugging Face `owkin/phikon-v2` | open | 224 | ImageNet | class token | 1024 |
| `hibou_l` | Hibou-L | Hugging Face `histai/hibou-L` | gated | 224 | mean 0.7068, 0.5755, 0.7220; std 0.1950, 0.2316, 0.1816 | pooler output | 1024 |
| `virchow2` | Virchow2 | Hugging Face `paige-ai/Virchow2` (timm) | gated | 224 | ImageNet | class token concatenated with the mean patch token | 2560 |
| `uni` | UNI2 | Hugging Face `MahmoodLab/UNI2-h` (timm) | gated | 224 | ImageNet | class token | 1536 |
| `hoptimus` | H-optimus-0 | Hugging Face `bioptimus/H-optimus-0` (timm) | gated | 224 | mean 0.7072, 0.5787, 0.7036; std 0.2119, 0.2301, 0.1775 | class token | 1536 |
| `midnight` | Midnight | Hugging Face `kaiko-ai/midnight` | open | 224 | mean 0.5; std 0.5 | class token concatenated with the mean patch token | 3072 |
| `dinov2_nat` | DINOv2-nat | Hugging Face `facebook/dinov2-base` | open | 224 | ImageNet | class token | 768 |
| `imagenet_vit` | ImageNet-ViT | timm `vit_base_patch16_224.augreg2_in21k_ft_in1k` | open | 224 | mean 0.5; std 0.5 | class token | 768 |

ImageNet normalisation is mean 0.485, 0.456, 0.406 and standard deviation 0.229, 0.224, 0.225.
CLIP normalisation is mean 0.4815, 0.4578, 0.4082 and standard deviation 0.2686, 0.2613, 0.2758.
The exact values are in `NORM` in `src/pipeline/export_embeddings.py`. The last two encoders
are general-vision baselines; the other eleven are pathology encoders.

Notes:

- Hibou-B and Hibou-L are loaded with `trust_remote_code=True`, which executes model code
  downloaded from their Hugging Face repositories.
- The CTransPath checkpoint is not downloaded automatically. It is expected at
  `./model_lib/pretrained/ctranspath.pth`; another location can be given with
  `--ctranspath-weights`.
- The CONCH package is installed by `uv sync` from its GitHub repository
  (https://github.com/Mahmoodlab/CONCH) at a pinned commit; its weights are downloaded from
  the Hugging Face repository.
- Weights that do not match the model definition raise an error at load time.

## Data

Six datasets define seven tasks. All data are placed under `./data`; the folder names below
are those read by `task_specs()` in `src/pipeline/export_embeddings.py`.

### NCT-CRC-HE-100K

Source: https://zenodo.org/records/1214456, file `NCT-CRC-HE-100K-NONORM.zip` (the version
without stain normalisation). No preparation script is needed. Expected layout:

```
data/nct/NCT-CRC-HE-100K-NONORM/{ADI,BACK,DEB,LYM,MUC,MUS,NORM,STR,TUM}/
```

### LC25000

Source: https://github.com/tampapath/lung_colon_image_set. The lung images (three classes)
and the colon images (two classes) are two separate tasks. No preparation script is needed.
Expected layout:

```
data/lc25000/lung_colon_image_set/lung_image_sets/{lung_n,lung_aca,lung_scc}/
data/lc25000/lung_colon_image_set/colon_image_sets/{colon_n,colon_aca}/
```

### BreaKHis

Source: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/.
Images of all magnifications are used, with the two classes benign and malignant. The patient
id is parsed from the file name. No preparation script is needed. Expected layout:

```
data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/{benign,malignant}/
```

### PatchCamelyon

Source: https://zenodo.org/records/2546921. Only the test split is used. The three files
`camelyonpatch_level_2_split_test_x.h5`, `camelyonpatch_level_2_split_test_y.h5` (both
decompressed) and `camelyonpatch_level_2_split_test_meta.csv` are placed under
`data/pcam_raw/pcam/`. Preparation:

```sh
uv run python src/data/prepare_pcam.py
```

This writes the patches as PNG files named by the slide they were cut from:

```
data/pcam/{normal,tumour}/<wsi>__<index>.png
```

### SICAPv2

Source: https://data.mendeley.com/datasets/9xxm58dvs3. The release is extracted to
`data/sicap_orig/SICAPv2` (containing `images/`, `partition/` and `wsi_labels.xlsx`).
All labelled patches are used, with the four classes NC, G3, G4 and G5 given by the primary
Gleason pattern. Preparation:

```sh
uv run python src/data/prepare_sicap.py --src data/sicap_orig/SICAPv2
```

This writes symbolic links to the patches and the slide-to-patient table:

```
data/sicap/{NC,G3,G4,G5}/<slide>_Block_Region_..._.jpg
data/sicap/slides.csv            slide_id, patient_id
```

### PANDA

Source: the Kaggle competition `prostate-cancer-grade-assessment`
(https://www.kaggle.com/competitions/prostate-cancer-grade-assessment). Kaggle API
credentials and acceptance of the competition rules are required. The file `train.csv` of the
competition is placed at `data/panda_raw/train.csv`. Preparation:

```sh
uv run python src/data/select_panda_slides.py \
    --train-csv data/panda_raw/train.csv --out data/panda_raw/slides.csv
uv run python src/data/download_panda.py \
    --csv data/panda_raw/slides.csv --out data/panda_raw/train_images
uv run python src/data/tile_panda.py \
    --csv data/panda_raw/slides.csv --images data/panda_raw/train_images \
    --out data/panda_tiles
```

- `select_panda_slides.py` draws 60 slides at random (seed 42) from each combination of data
  provider and ISUP grade, 720 slides in total, and writes `data/panda_raw/slides.csv`.
- `download_panda.py` downloads the selected whole-slide images to
  `data/panda_raw/train_images/`. Slides already present are skipped, so the script can be
  run again until it reports no failures.
- `tile_panda.py` cuts non-overlapping tiles of 224 x 224 pixels at pyramid level 1, keeps a
  tile when at least 35% of it is tissue, and keeps at most 100 randomly drawn tiles per
  slide. Every tile carries the ISUP grade of its slide. Output:

```
data/panda_tiles/{0,1,2,3,4,5}/<image_id>_<k>.png
data/panda_tiles/slides.csv      image_id, data_provider, isup_grade, n_tiles
```

### Patch selection and splits

For each task the export lists the image files in sorted order and removes exact duplicates
(images with identical pixels within the task; the first occurrence is kept). NCT-CRC-HE-100K
and PatchCamelyon are then reduced to a class-stratified sample of 9000 patches; the other
tasks use every remaining patch. One fifth of each task is held out as the test split,
stratified by class. For tasks with patient or slide identifiers, one fifth of the patients or
slides is held out and all patches of a patient or slide stay on the same side. The PANDA
split is stratified by grade and data provider jointly.

| Task key | Dataset | Organ | Classes | Patches used | Group unit of the split |
|---|---|---|---|---|---|
| `nct` | NCT-CRC-HE-100K | colorectum | 9 | 9000 | patch |
| `lung` | LC25000 lung | lung | 3 | 14195 | patch |
| `lc_colon` | LC25000 colon | colon | 2 | 9525 | patch |
| `breakhis` | BreaKHis | breast | 2 | 7784 | patient (70) |
| `pcam` | PatchCamelyon | lymph node | 2 | 9000 | slide (129) |
| `sicap` | SICAPv2 | prostate | 4 | 12081 | patient (95) |
| `panda` | PANDA | prostate | 6 | 27075 | slide (720) |

BreaKHis patches are grouped by the biopsy number in their file name; identifiers that share
the number and differ only in a letter suffix form one group.

The SICAPv2 and PANDA labels are ordinal (NC < G3 < G4 < G5 and ISUP grade 0 to 5); the other
five tasks are categorical.

## Pipeline

1. Embedding export (GPU or MPS). One file `<model>__<task>.npz` is written per encoder and
   task; existing files are skipped.

   ```sh
   uv run python src/pipeline/export_embeddings.py --base ./data --out ./embeddings
   ```

   The options `--models` and `--tasks` restrict the export; `--batch` and `--workers` set
   the batch size and the number of data-loading processes.

2. Analysis (CPU). All stages are run in order from the cached embeddings in `./embeddings`;
   the result files are written to `./results`, the figures to `./results/figures` and the
   tables to `./results/tables`.

   ```sh
   sh scripts/run_pipeline.sh
   ```

Each stage can also be run on its own. The stages that read embeddings accept `--emb-dir`,
`--out-dir` and `--models`; `ood_summary.py`, `make_figures.py` and `make_tables.py` read result
files from `--res-dir` and write to `--out-dir` (by default the same directory; its `tables`
subdirectory for `make_tables.py`).

| Stage | Computes | Outputs |
|---|---|---|
| `src/pipeline/ood_sweep.py` | AUROC of the four scorers for every out-of-distribution condition of every encoder and task, on the stored split. | `ood_table.csv`, `probe_stored.csv` |
| `src/analysis/probe_metrics.py` | Accuracy, expected calibration error, AURC, E-AURC and, for the ordinal tasks, quadratic weighted kappa of the probe over five splits, with bootstrap intervals. | `probe_splits.csv`, `probe_summary.csv`, `probe_metrics.md`, `selective_curves.npz` |
| `src/analysis/ood_summary.py` | Summary tables of the AUROC per encoder, tier and scorer, and Spearman correlations across encoders between probe accuracy and AUROC. | `ood_summary.md`, `correlations.csv` |
| `src/analysis/blur_accuracy.py` | Accuracy of the probe on the test split and on the same patches under Gaussian blur, on the stored split. | `blur_accuracy.csv`, `blur_accuracy.md` |
| `src/analysis/probe_sensitivity.py` | Probe metrics on the tasks with patient or slide ids with the regularisation strength fixed at each grid value and under four selection protocols, and the reliability table of the probe on PANDA. | `probe_sensitivity_fixed_c.csv`, `probe_sensitivity_protocols.csv`, `probe_reliability_panda.csv`, `probe_sensitivity.md` |
| `src/analysis/scorer_sensitivity.py` | AUROC of the Mahalanobis and kNN scorers on the far and blur conditions under different feature normalisation, neighbourhood size and covariance estimate. | `scorer_sensitivity.csv`, `scorer_sensitivity.md` |
| `src/analysis/gate_operating_point.py` | Rejection rates, coverage and accuracy of the two-threshold abstention rule with thresholds set on a validation part of the training split. | `gate_operating_point.csv`, `gate_operating_point.md` |
| `src/analysis/slide_aggregation.py` | Slide-level ISUP grading on PANDA with five aggregators of the patch embeddings over five slide-grouped splits. | `slide_aggregation.csv`, `slide_aggregation.md`, `figures/fig_slide_aggregation.png` |
| `src/analysis/label_coarsening.py` | Probe metrics on SICAPv2 with region-level patch labels and with every patch relabelled by the most severe grade of its slide, and an attention aggregator trained on slide bags. | `label_coarsening.csv`, `label_coarsening.md` |
| `src/analysis/sicap_without_benign.py` | Probe metrics on SICAPv2 with the benign class removed from the training and test splits, and the number of patients that contribute benign patches. | `sicap_without_benign.csv`, `sicap_without_benign.md` |
| `src/figures/make_figures.py` | Figures drawn from the result files; nothing is fitted. | `figures/fig_accuracy_vs_ood.png`, `figures/fig_blur.png`, `figures/fig_selective_accuracy.png` |
| `src/figures/make_tables.py` | LaTeX tables written from the result files; nothing is fitted. | `tables/tab_tiers.tex`, `tables/tab_grading.tex`, `tables/tab_slide.tex`, `tables/tab_gate.tex`, `tables/tab_blur.tex` |

`ood_summary.py` reads the outputs of `ood_sweep.py` and `probe_metrics.py`, and
`make_figures.py` reads those of the first three stages. `make_tables.py` reads
`ood_table.csv`, `probe_summary.csv`, `gate_operating_point.csv`, `slide_aggregation.csv` and
`blur_accuracy.csv`,
and needs the results of all thirteen encoders. The neural aggregators of
`slide_aggregation.py` and `label_coarsening.py` are trained on CPU by default (`--device`).

## Method in brief

### Probe

Embeddings are standardised and classified with a multinomial logistic regression. The
inverse regularisation strength C is selected over seven log-spaced values from 1e-6 to 1 by
three-fold cross-validation on the training split, using the log-loss of the held-out folds.
For tasks with patient or slide identifiers the folds keep each patient or slide on one side.
The test split takes no part in the selection. Confidence is the maximum softmax probability
of the probe.

### Scorers

Four post-hoc novelty scorers are fitted on the training split only. A higher score means
more out-of-distribution.

| Key | Definition |
|---|---|
| `maha` | Squared Mahalanobis distance to the nearest class mean on raw embeddings, with one pooled within-class covariance estimated with Ledoit-Wolf shrinkage. This is the primary scorer. |
| `knn` | Euclidean distance to the 50th nearest training embedding after L2 normalisation. |
| `energy` | Negative log-sum-exp of the probe logits. |
| `msp` | One minus the maximum softmax probability of the probe. |

### Out-of-distribution tiers and conditions

The conditions are defined in `src/lib/conditions.py`.

| Tier | Condition name | In-distribution set | Out-of-distribution set |
|---|---|---|---|
| far | `far_<donor>` | test split of the task | test split of a donor task embedded with the same encoder (`FAR_OOD` in `src/lib/config.py`) |
| blur | `blur_s<sigma>` | test split of the task | the same test patches under Gaussian blur with sigma 1, 2 or 3 pixels of a 224 x 224 input (categorical tasks) |
| near | `holdout_<class>` | test patches of the remaining classes | test patches of one class that is removed from the training split (tasks with three or more classes) |
| artefact | `artefact` | test patches of the remaining classes | test patches of the NCT-CRC-HE-100K classes BACK, DEB and ADI, removed jointly from the training split |

In the near and artefact tiers the probe and the scorers are refitted on the reduced training
split. The summary tables separate the near tier into categorical and grading tasks.

### Metrics

- AUROC of a novelty score, with the out-of-distribution set as the positive class.
- Accuracy and, for the ordinal tasks, quadratic weighted kappa (QWK).
- Top-label expected calibration error (ECE) with 15 equal-width confidence bins.
- Area under the risk-coverage curve (AURC), abstaining on the least confident predictions
  first, and its excess over a perfect ranking at the same error rate (E-AURC).
- Selective accuracy: accuracy of the most confident fraction of the predictions.

### Splits and confidence intervals

The exported files store one split (seed 42), which is used by the detection sweep, the blur
accuracy, the scorer sensitivity analysis and the abstention rule. The probe metrics, the probe sensitivity
analysis, the slide-level aggregation, the label-coarsening analysis and the SICAPv2 analysis
without the benign class pool the stored
embeddings and draw five further splits (seeds 0 to 4) in the same way.

Within each split the test patches are resampled 300 times. For tasks with patient or slide
identifiers the resampling is a cluster bootstrap: patients or slides are drawn with
replacement and all their patches are taken. For the other tasks patches are drawn
individually. The point estimate is the mean of the metric over the splits and the 95%
interval is given by the 2.5th and 97.5th percentiles of the bootstrap draws pooled over the
splits. Intervals of the Spearman correlations are percentile bootstrap intervals over
encoders (2000 resamples).

### Abstention rule

The rule is evaluated on the stored split by `src/analysis/gate_operating_point.py`.

1. The training split is divided into a fit part and a validation part of one fifth of the
   patches (or of the patients or slides), stratified in the same way as the test split.
2. The Mahalanobis scorer and the probe are fitted on the fit part.
3. The novelty threshold is the 95th percentile of the Mahalanobis scores of the validation
   patches. A patch is rejected when its score exceeds this threshold.
4. The confidence threshold is the quantile of the probe confidence, among the validation
   patches that pass the novelty threshold, that retains 80% of them. A patch is retained
   when its confidence is at least this threshold.
5. Both thresholds are applied unchanged to the test split and to the far, blur and artefact
   out-of-distribution sets.

## Repository layout

```
scripts/
  run_pipeline.sh            runs every analysis stage in order
src/
  data/
    prepare_pcam.py          writes the PatchCamelyon test split as class folders
    prepare_sicap.py         arranges the labelled SICAPv2 patches as class folders
    select_panda_slides.py   selects the PANDA slides by provider and ISUP grade
    download_panda.py        downloads the selected PANDA slides from Kaggle
    tile_panda.py            tiles the PANDA slides into patches
  pipeline/
    export_embeddings.py     embeds every task with every encoder and caches the result
    ood_sweep.py             detection AUROC of the four scorers on all conditions
  analysis/
    probe_metrics.py         probe accuracy, calibration and selective prediction
    ood_summary.py           summary tables and rank correlations of the detection sweep
    blur_accuracy.py         probe accuracy on the blurred test split
    probe_sensitivity.py     sensitivity to the regularisation and selection of the probe
    scorer_sensitivity.py    sensitivity to the settings of the feature-space scorers
    gate_operating_point.py  the two-threshold abstention rule
    slide_aggregation.py     slide-level aggregation on PANDA
    label_coarsening.py      region-level and slide-level labels on SICAPv2
    sicap_without_benign.py  SICAPv2 with the benign class removed
  figures/
    make_figures.py          figures drawn from the result files
    make_tables.py           LaTeX tables written from the result files
  lib/
    config.py                encoder and task names, constants, paths
    names.py                 display names of the tasks
    splits.py                train/test splitting and slide identifiers
    probe.py                 the linear probe
    scoring.py               novelty scorers and metrics
    conditions.py            out-of-distribution tiers, condition names and sets
    evaluation.py            probe predictions and metrics of a set of predictions
    resampling.py            bootstrap resampling and pooled intervals
    tables.py                writing of csv and markdown result files
```

## Licences and attribution

The code in this repository is released under the MIT licence (see `LICENSE`). This licence
covers the code only, not the datasets or the model weights, which keep their own terms.

The repository redistributes no data, model weights or embeddings. The licences below are
those stated on the public page of each source; the terms of each source apply to its use.

| Dataset | Licence |
|---|---|
| NCT-CRC-HE-100K | CC BY 4.0 (Zenodo record 1214456) |
| LC25000 | see source |
| BreaKHis | CC BY 4.0, for non-commercial research with citation (dataset page) |
| PatchCamelyon | CC0 (https://github.com/basveeling/pcam) |
| SICAPv2 | CC BY 4.0 (Mendeley Data) |
| PANDA | see source |

| Model | Licence |
|---|---|
| Lunit (`1aurent/vit_small_patch8_224.lunit_dino`) | see source |
| CTransPath | see source |
| CONCH | CC BY-NC-ND 4.0 |
| Hibou-B, Hibou-L | Apache 2.0 |
| Phikon, Phikon-v2 | see source |
| Virchow2 | CC BY-NC-ND 4.0 |
| UNI2 (`MahmoodLab/UNI2-h`) | CC BY-NC-ND 4.0 |
| H-optimus-0 | Apache 2.0 |
| Midnight | MIT |
| DINOv2-nat (`facebook/dinov2-base`) | Apache 2.0 |
| ImageNet-ViT (`timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`) | Apache 2.0 |

The convolutional patch-embedding module of CTransPath in `src/pipeline/export_embeddings.py`
follows the reference implementation at https://github.com/Xiyue-Wang/TransPath.

## Citation

The accompanying paper, "A Retraining-Free Reliability Audit and Abstention Gate for Frozen
Pathology Foundation Models", is under review. A reference will be
added on publication.
