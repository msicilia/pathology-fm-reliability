"""
export_embeddings.py
====================
Embed every patch of every task with every frozen encoder and cache the result.
All analyses run on these cached vectors; no encoder is trained or fine-tuned.

For each task the patches are listed in sorted order, exact duplicates (identical
pixels) are removed, large tasks are reduced to a class-stratified budget, and
one train/test split is drawn (see src/lib/splits.py: stratified by class, and by
patient or slide where the task provides them).

Each encoder is loaded with all of its released weights (a mismatch raises),
fed its own input size and normalisation, and pooled as its authors recommend
for linear probing (`extract_features`). Images are resized on the shorter side
with bicubic interpolation and centre-cropped; no augmentation is applied.

Output, one file per (model, task):
    <out>/<model>__<task>.npz
        Z_train (N x D) float32, y_train (N,) int, paths_train (N,) file names
        Z_test, y_test, paths_test
        groups_train, groups_test    patient or slide id per patch (grouped tasks)
        sites_train, sites_test      data provider per patch (PANDA)
        classes (C,)                 class names in label order
        Z_test_blur_s<sigma>         test embeddings under Gaussian blur
                                     (categorical tasks)

Blur is a Gaussian filter applied to the network input; sigma is expressed in
pixels of a 224x224 input and scaled for encoders with a larger input.

Run from the repository root:
    uv run python src/pipeline/export_embeddings.py --base ./data --out ./embeddings
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from src.lib.config import BLUR_SIGMAS, CATEGORICAL, ENCODERS, EXPORT_SEED, TASKS
from src.lib.splits import joint_strata, split_indices

Image.MAX_IMAGE_PIXELS = None
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    root: str                                   # folder under --base
    classes: dict[str, str]                     # class folder -> class name, in label order
    limit: int = 0                              # class-stratified patch budget (0 = all)
    group: Callable[[str], str] | None = None   # file name -> patient or slide id
    site: Callable[[str], str] | None = None    # file name -> data provider
    sidecar: str | None = None                  # csv under root used by group/site
    table: dict = field(default_factory=dict)


def _breakhis_patient(name: str) -> str:
    # SOB_<B|M>_<subtype>-<year>-<biopsy><letters>-<magnification>-<index>.png
    # Ids that share the biopsy number and differ in the letter suffix are grouped.
    m = re.match(r"SOB_[BM]_[A-Z]+-(\d+)-(\d+)[A-Za-z]*-\d+-\d+\.png$", name)
    if m is None:
        raise ValueError(f"unexpected BreaKHis file name: {name}")
    return f"{m.group(1)}-{m.group(2)}"


def _pcam_slide(name: str) -> str:          # <wsi>__<index>.png
    return name.split("__")[0]


def _panda_slide(name: str) -> str:         # <image_id>_<k>.png
    return name.rsplit("_", 1)[0]


def _sicap_slide(name: str) -> str:         # <slide_id>_Block_Region_..._.jpg
    return name.split("_Block")[0]


def task_specs() -> dict[str, Task]:
    def same(*names):
        return {n: n for n in names}

    panda = Task("panda_tiles", same("0", "1", "2", "3", "4", "5"), sidecar="slides.csv")
    sicap = Task("sicap", same("NC", "G3", "G4", "G5"), sidecar="slides.csv")

    def panda_provider(name):
        return panda.table["data_provider"][_panda_slide(name)]

    def sicap_patient(name):
        return str(sicap.table["patient_id"][_sicap_slide(name)])

    panda.group = _panda_slide
    panda.site = panda_provider
    sicap.group = sicap_patient
    return {
        "nct": Task("nct/NCT-CRC-HE-100K-NONORM",
                    same("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"),
                    limit=9000),
        "lung": Task("lc25000/lung_colon_image_set/lung_image_sets",
                     {"lung_n": "lung_benign", "lung_aca": "lung_aca", "lung_scc": "lung_scc"}),
        "lc_colon": Task("lc25000/lung_colon_image_set/colon_image_sets",
                         {"colon_n": "colon_benign", "colon_aca": "colon_aca"}),
        "breakhis": Task("breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast",
                         same("benign", "malignant"), group=_breakhis_patient),
        "pcam": Task("pcam", same("normal", "tumour"), limit=9000, group=_pcam_slide),
        "sicap": sicap,
        "panda": panda,
    }


def _pixel_hash(path: str) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        return hashlib.md5(img.tobytes() + str(img.size).encode()).hexdigest()


def scan(root: str, classes: dict[str, str]):
    """List the image files of each class folder, in sorted order."""
    paths, labels = [], []
    for label, folder in enumerate(classes):
        found = []
        for d, _, files in os.walk(os.path.join(root, folder), followlinks=True):
            found += [os.path.join(d, f) for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]
        if not found:
            raise FileNotFoundError(f"no images under {os.path.join(root, folder)}")
        paths += sorted(found)
        labels += [label] * len(found)
    return np.array(paths), np.array(labels)


def load_task(base: str, name: str, spec: Task, workers: int) -> dict:
    """Scan one task, drop duplicate images, apply its budget and split it."""
    root = os.path.join(base, spec.root)
    paths, labels = scan(root, spec.classes)
    if spec.sidecar:
        table = pd.read_csv(os.path.join(base, spec.root.split("/")[0], spec.sidecar))
        spec.table = table.set_index(table.columns[0]).to_dict()

    with ProcessPoolExecutor(workers) as pool:
        hashes = list(tqdm(pool.map(_pixel_hash, paths, chunksize=256), total=len(paths),
                           desc=f"{name} hashing", leave=False))
    _, first = np.unique(hashes, return_index=True)
    keep = np.sort(first)
    n_dup = len(paths) - len(keep)
    paths, labels = paths[keep], labels[keep]

    if spec.limit and len(paths) > spec.limit:
        paths, _, labels, _ = train_test_split(paths, labels, train_size=spec.limit,
                                               random_state=EXPORT_SEED, stratify=labels)
        order = np.argsort(paths)
        paths, labels = paths[order], labels[order]

    names = np.array([os.path.basename(p) for p in paths])
    if len(set(names)) != len(names):
        raise RuntimeError(f"{name}: file names are not unique")
    groups = None if spec.group is None else np.array([spec.group(n) for n in names])
    sites = None if spec.site is None else np.array([spec.site(n) for n in names])
    tr, te = split_indices(labels, groups, joint_strata(labels, sites), EXPORT_SEED)
    if groups is not None and set(groups[tr]) & set(groups[te]):
        raise RuntimeError(f"{name}: a group is on both sides of the split")

    print(f"  {name}: {len(tr)} train / {len(te)} test, {len(spec.classes)} classes, "
          f"{n_dup} duplicates removed"
          + ("" if groups is None else f", {len(set(groups[tr]))} / {len(set(groups[te]))} groups"))
    return dict(paths=paths, labels=labels, names=names, groups=groups, sites=sites,
                train=tr, test=te, classes=list(spec.classes.values()))


# --------------------------------------------------------------------------- #
# encoders: preprocessing, weights and pooling
# --------------------------------------------------------------------------- #
_IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
_CLIP = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
_HALF = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
_HIBOU = ([0.7068, 0.5755, 0.7220], [0.1950, 0.2316, 0.1816])

# Input normalisation of each encoder, as published with its weights.
NORM = {
    "uni": _IMAGENET,
    "virchow2": _IMAGENET,
    "phikon": _IMAGENET,
    "phikon_v2": _IMAGENET,
    "ctranspath": _IMAGENET,
    "conch": _CLIP,
    "hibou_b": _HIBOU,
    "hibou_l": _HIBOU,
    "hoptimus": ([0.707223, 0.578729, 0.703617], [0.211883, 0.230117, 0.177517]),
    "lunit": ([0.70322989, 0.53606487, 0.66096631], [0.21716536, 0.26081574, 0.20723464]),
    "midnight": _HALF,
    "dinov2_nat": _IMAGENET,
    "imagenet_vit": _HALF,
}
INPUT_SIZE = {"conch": 448}     # all other encoders take 224x224


def input_size(name: str) -> int:
    return INPUT_SIZE.get(name, 224)


def eval_transform(name: str):
    size = input_size(name)
    mean, std = NORM[name]
    return transforms.Compose([
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class ConvStem(nn.Module):
    """Convolutional patch embedding of CTransPath (Wang et al., 2022), following the
    reference implementation at https://github.com/Xiyue-Wang/TransPath."""

    def __init__(self, embed_dim: int = 96):
        super().__init__()
        stem, in_dim, out_dim = [], 3, embed_dim // 8
        for _ in range(2):
            stem += [nn.Conv2d(in_dim, out_dim, 3, 2, 1, bias=False),
                     nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)]
            in_dim, out_dim = out_dim, out_dim * 2
        stem.append(nn.Conv2d(in_dim, embed_dim, 1))
        self.proj = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x).permute(0, 2, 3, 1))     # B, H, W, C


def _build_ctranspath(weights: str):
    import timm
    if not os.path.exists(weights):
        raise FileNotFoundError(f"CTransPath checkpoint not found at {weights}")
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False,
                              embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])
    model.patch_embed = ConvStem(embed_dim=96)
    model.head = nn.Identity()
    state = torch.load(weights, map_location="cpu")
    state = state.get("model", state)
    # The checkpoint places each patch-merging layer at the end of a stage
    # (layers.{0,1,2}.downsample); timm 1.x places it at the start of the next one.
    remapped = {}
    for key, value in state.items():
        m = re.match(r"layers\.(\d+)\.downsample\.(.*)", key)
        remapped[f"layers.{int(m.group(1)) + 1}.downsample.{m.group(2)}" if m else key] = value
    result = model.load_state_dict(remapped, strict=False)
    # relative_position_index and attn_mask are buffers recomputed by timm
    buffers = ("relative_position_index", "attn_mask")
    missing = [k for k in result.missing_keys if not any(b in k for b in buffers)]
    unexpected = [k for k in result.unexpected_keys if not any(b in k for b in buffers)]
    if missing or unexpected:
        raise RuntimeError(f"CTransPath weights do not match: missing={missing} "
                           f"unexpected={unexpected}")
    return model


def _build_conch():
    from conch.open_clip_custom import create_model_from_pretrained
    from huggingface_hub import hf_hub_download
    weights = hf_hub_download(repo_id="MahmoodLab/conch", filename="pytorch_model.bin")
    model = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=weights,
                                         return_transform=False)
    state = torch.load(weights, map_location="cpu")
    visual = {k[len("visual."):]: v for k, v in state.items() if k.startswith("visual.")}
    model.visual.load_state_dict(visual, strict=True)
    return model.visual


def _hf_model(repo: str, trust_remote_code: bool = False):
    from transformers import AutoModel
    model, info = AutoModel.from_pretrained(repo, trust_remote_code=trust_remote_code,
                                            output_loading_info=True)
    if info["missing_keys"] or info["unexpected_keys"] or info["mismatched_keys"]:
        raise RuntimeError(f"{repo} weights do not match: {info}")
    return model


def build_backbone(name: str, ctranspath_weights: str):
    """Load one frozen encoder. timm loads its hub checkpoints strictly."""
    import timm
    from timm.layers import SwiGLUPacked
    if name == "uni":           # configuration of the MahmoodLab/UNI2-h model card
        return timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, img_size=224, patch_size=14,
            depth=24, num_heads=24, init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
            num_classes=0, no_embed_class=True, mlp_layer=SwiGLUPacked, act_layer=nn.SiLU,
            reg_tokens=8, dynamic_img_size=True)
    if name == "virchow2":      # configuration of the paige-ai/Virchow2 model card
        return timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                                 mlp_layer=SwiGLUPacked, act_layer=nn.SiLU)
    if name == "hoptimus":      # configuration of the bioptimus/H-optimus-0 model card
        return timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True,
                                 init_values=1e-5, dynamic_img_size=False)
    if name == "lunit":         # DINO ViT-S/8 of Kang et al. (2023), timm port
        return timm.create_model("hf-hub:1aurent/vit_small_patch8_224.lunit_dino",
                                 pretrained=True, num_classes=0)
    if name == "imagenet_vit":  # supervised ViT-B/16, ImageNet-21k then ImageNet-1k
        return timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k",
                                 pretrained=True, num_classes=0)
    if name == "phikon":
        return _hf_model("owkin/phikon")
    if name == "phikon_v2":
        return _hf_model("owkin/phikon-v2")
    if name == "midnight":
        return _hf_model("kaiko-ai/midnight")
    if name == "hibou_b":
        return _hf_model("histai/hibou-b", trust_remote_code=True)
    if name == "hibou_l":
        return _hf_model("histai/hibou-L", trust_remote_code=True)
    if name == "dinov2_nat":    # DINOv2 ViT-B/14 trained on natural images
        return _hf_model("facebook/dinov2-base")
    if name == "conch":
        return _build_conch()
    if name == "ctranspath":
        return _build_ctranspath(ctranspath_weights)
    raise ValueError(name)


@torch.no_grad()
def extract_features(backbone, name: str, x):
    """One embedding per patch, pooled as recommended for each encoder.

    class token:                      UNI2, H-optimus-0, Lunit, ImageNet-ViT,
                                      Phikon, Phikon-v2, DINOv2-nat, Hibou-B, Hibou-L
    class token + mean patch token:   Virchow2 (2560-d), Midnight (3072-d)
    attentional pooler, no projection: CONCH (512-d)
    mean over the final feature map:  CTransPath (768-d)
    """
    if name in ("uni", "hoptimus", "lunit", "imagenet_vit"):
        feat = backbone(x)
    elif name in ("phikon", "phikon_v2", "dinov2_nat"):
        feat = backbone(pixel_values=x).last_hidden_state[:, 0]
    elif name in ("hibou_b", "hibou_l"):
        feat = backbone(pixel_values=x).pooler_output
    elif name == "virchow2":        # tokens: class, 4 registers, 256 patches
        out = backbone(x)
        feat = torch.cat([out[:, 0], out[:, 5:].mean(1)], dim=-1)
    elif name == "midnight":        # tokens: class, 256 patches
        out = backbone(pixel_values=x).last_hidden_state
        feat = torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=-1)
    elif name == "conch":
        feat = backbone.forward_no_head(x, normalize=False)
    elif name == "ctranspath":
        feat = backbone.forward_features(x).mean(dim=[1, 2])
    else:
        raise ValueError(name)
    if feat.dim() != 2:
        raise RuntimeError(f"{name}: expected one vector per patch, got {tuple(feat.shape)}")
    return feat.float()


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #
class PatchDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            return self.transform(img.convert("RGB"))


@torch.no_grad()
def embed(backbone, name: str, paths, device, batch: int, workers: int, sigmas=()):
    """Embed `paths`; also return the embeddings of the blurred inputs per sigma."""
    loader = DataLoader(PatchDataset(paths, eval_transform(name)), batch_size=batch,
                        shuffle=False, num_workers=workers)
    scale = input_size(name) / 224
    unblurred, blurred = [], {s: [] for s in sigmas}
    for x in tqdm(loader, desc=name, leave=False):
        x = x.to(device)
        unblurred.append(extract_features(backbone, name, x).cpu().numpy())
        for s in sigmas:
            sigma = s * scale
            kernel = max(3, int(6 * sigma) | 1)
            xb = TF.gaussian_blur(x, kernel_size=kernel, sigma=sigma)
            blurred[s].append(extract_features(backbone, name, xb).cpu().numpy())
    return (np.concatenate(unblurred).astype(np.float32),
            {s: np.concatenate(v).astype(np.float32) for s, v in blurred.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="./data", help="dataset root")
    ap.add_argument("--out", default="./embeddings")
    ap.add_argument("--models", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--ctranspath-weights", default="./model_lib/pretrained/ctranspath.pth")
    ap.add_argument("--blur", nargs="*", type=float, default=list(BLUR_SIGMAS),
                    help="blur sigmas for the categorical tasks")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}  out={args.out}")

    specs = task_specs()
    tasks = {t: load_task(args.base, t, specs[t], args.workers) for t in args.tasks}

    for m in args.models:
        todo = [t for t in tasks if not os.path.exists(os.path.join(args.out, f"{m}__{t}.npz"))]
        if not todo:
            continue
        print(f"=== {m} ===")
        backbone = build_backbone(m, args.ctranspath_weights).to(device).eval()
        for t in todo:
            task = tasks[t]
            tr, te = task["train"], task["test"]
            sigmas = tuple(args.blur) if t in CATEGORICAL else ()
            z_train, _ = embed(backbone, m, task["paths"][tr], device, args.batch, args.workers)
            z_test, z_blur = embed(backbone, m, task["paths"][te], device, args.batch,
                                   args.workers, sigmas)
            payload = dict(Z_train=z_train, y_train=task["labels"][tr],
                           paths_train=task["names"][tr],
                           Z_test=z_test, y_test=task["labels"][te],
                           paths_test=task["names"][te],
                           classes=np.array(task["classes"]))
            for key in ("groups", "sites"):
                if task[key] is not None:
                    payload[f"{key}_train"] = task[key][tr]
                    payload[f"{key}_test"] = task[key][te]
            for s, z in z_blur.items():
                payload[f"Z_test_blur_s{s}"] = z
            out = os.path.join(args.out, f"{m}__{t}.npz")
            partial = out + ".partial.npz"       # written in full, then renamed
            np.savez_compressed(partial, **payload)
            os.replace(partial, out)
            print(f"  wrote {out}  (D={z_train.shape[1]})")
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
