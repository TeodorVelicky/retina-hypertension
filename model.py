"""Shared model code for the retinal hypertension project.

Used by both the Kaggle training pipeline (upgrade.py) and the Hugging Face Space (app.py),
so features are computed the same way in both places.
"""
import glob
import math

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

SIZE = 448                                            # every image is prepared at 448 x 448
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet statistics (DINOv2 and RETFound)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DINO = "vit_base_patch14_dinov2.lvd142m"
RETFOUND_REPO = "YukunZhou/RETFound_mae_natureCFP"

# feature name -> (backbone key, pooling). Backbone keys: d = DINOv2 ViT-B/14, r = RETFound ViT-L/16
FEATURES = {
    "d224_cls": ("d224", "cls"),
    "d448_cls": ("d448", "cls"),
    "d448_mean": ("d448", "mean"),
    "r224_cls": ("r224", "cls"),
    "r224_mean": ("r224", "mean"),
    "r448_cls": ("r448", "cls"),
    "r448_mean": ("r448", "mean"),
}
RES = {"d224": 224, "d448": 448, "r224": 224, "r448": 448}


# ---------------------------------------------------------------- preprocessing
def crop_black_border(img, threshold=10):
    """Crop to the bounding box of non-black pixels, then pad to a square."""
    a = np.array(img)
    mask = a.max(axis=2) > threshold
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    a = a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = a.shape[:2]
    d = abs(h - w)
    if h < w:
        a = np.pad(a, ((d // 2, d - d // 2), (0, 0), (0, 0)))
    elif w < h:
        a = np.pad(a, ((0, 0), (d // 2, d - d // 2), (0, 0)))
    return Image.fromarray(a)


def base_image(img):
    """Any fundus photo -> cropped, square, 448 x 448 RGB PIL image."""
    return crop_black_border(img.convert("RGB")).resize((SIZE, SIZE), Image.BICUBIC)


def clahe(img, clip=2.0, tiles=8):
    """Vessel-enhancing contrast: CLAHE on the lightness channel (LAB), colours kept."""
    lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def to_tensor(img):
    a = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(((a - MEAN) / STD).transpose(2, 0, 1).copy())


def prepare(img, use_clahe):
    """PIL image -> (display RGB uint8 array, normalised 3x448x448 tensor)."""
    b = base_image(img)
    x = to_tensor(clahe(b) if use_clahe else b)
    return np.asarray(b), x


# ---------------------------------------------------------------- backbones
def load_dino(img_size):
    m = timm.create_model(DINO, pretrained=True, num_classes=0, img_size=img_size)
    return m.eval()


def load_retfound_state(token=None):
    """Download the gated RETFound colour-fundus checkpoint and return its state dict."""
    from huggingface_hub import snapshot_download
    d = snapshot_download(RETFOUND_REPO, token=token)
    files = sorted(p for p in glob.glob(f"{d}/**/*", recursive=True)
                   if p.endswith((".pth", ".pt", ".bin", ".safetensors")))
    if not files:
        raise FileNotFoundError(f"no weight file found in {d}")
    path = files[0]
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path)
    else:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def build_retfound(state, img_size):
    """RETFound encoder (MAE ViT-L/16) in timm's ViT layout, every encoder weight loaded."""
    from timm.layers import resample_abs_pos_embed
    m = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0, img_size=img_size)
    own = m.state_dict()
    state = dict(state)
    if "norm.weight" not in state and "fc_norm.weight" in state:   # fine-tune style checkpoint
        state["norm.weight"], state["norm.bias"] = state["fc_norm.weight"], state["fc_norm.bias"]
    sd = {}
    for k, v in state.items():
        if k == "pos_embed" and k in own and v.shape != own[k].shape:
            g = img_size // 16
            v = resample_abs_pos_embed(v, new_size=(g, g), num_prefix_tokens=1)
        if k in own and v.shape == own[k].shape:
            sd[k] = v
    missing = [k for k in own if k not in sd]
    if missing:
        raise RuntimeError(f"RETFound weights missing for {len(missing)} keys, e.g. {missing[:6]}")
    m.load_state_dict(sd, strict=True)
    return m.eval()


def load_backbones(keys, token=None):
    models, state = {}, None
    for k in keys:
        if k.startswith("d"):
            models[k] = load_dino(RES[k])
        else:
            state = state if state is not None else load_retfound_state(token)
            models[k] = build_retfound(state, RES[k])
    return models


# ---------------------------------------------------------------- features
def token_features(models, x448, keys):
    """Normalised [B,3,448,448] batch -> {backbone key: token tensor [B, 1+N, D]}."""
    out, x224 = {}, None
    for k in keys:
        x = x448
        if RES[k] == 224:
            if x224 is None:
                x224 = F.interpolate(x448, size=(224, 224), mode="bicubic",
                                     align_corners=False, antialias=True)
            x = x224
        out[k] = models[k].forward_features(x)
    return out


def pooled(models, tokens, name):
    k, pool = FEATURES[name]
    t = tokens[k]
    if pool == "cls":
        return t[:, 0]
    return t[:, models[k].num_prefix_tokens:].mean(1)


class RetinaModel(nn.Module):
    """Backbone(s) -> chosen pooled features (optionally flip-averaged) -> one linear logit.

    The StandardScaler and logistic regression are folded into `head`:
    w' = w / sigma,  b' = b - sum(w * mu / sigma).
    """

    def __init__(self, models, feature_names, w, b, tta):
        super().__init__()
        self.backbones = nn.ModuleDict(models)
        self.feature_names = list(feature_names)
        self.tta = bool(tta)
        self.needed = sorted({FEATURES[n][0] for n in self.feature_names})
        self.head = nn.Linear(len(w), 1)
        with torch.no_grad():
            self.head.weight.copy_(torch.as_tensor(np.asarray(w), dtype=torch.float32)[None])
            self.head.bias.fill_(float(b))

    def features(self, x):
        tok = token_features(self.backbones, x, self.needed)
        return torch.cat([pooled(self.backbones, tok, n).float() for n in self.feature_names], 1)

    def forward(self, x, tta=None):
        tta = self.tta if tta is None else tta
        f = self.features(x)
        if tta:
            f = (f + self.features(torch.flip(x, dims=[3]))) / 2
        return self.head(f)


def fold_head(scaler, clf):
    w, b = clf.coef_[0], clf.intercept_[0]
    w_prime = w / scaler.scale_
    b_prime = b - np.sum(w * scaler.mean_ / scaler.scale_)
    return w_prime.astype(np.float32), float(b_prime)


# ---------------------------------------------------------------- Grad-CAM
class _NoTTA(nn.Module):
    """Grad-CAM needs one pass per layer; the flipped pass would overwrite the hooks."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x, tta=False)


def vit_reshape(t):
    n = t.shape[1]
    g = math.isqrt(n - 1)
    t = t[:, n - g * g:, :]                     # drop CLS (and any register) tokens
    return t.reshape(t.size(0), g, g, t.size(2)).permute(0, 3, 1, 2)


def make_cam(model):
    from pytorch_grad_cam import GradCAM
    for p in model.parameters():
        p.requires_grad_(True)                  # not training: autograd only tracks the path
    layers = [model.backbones[k].blocks[-1].norm1 for k in model.needed]
    return GradCAM(model=_NoTTA(model), target_layers=layers, reshape_transform=vit_reshape)


def cam_overlay(cam, x, rgb_uint8):
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    g = cam(input_tensor=x[None], targets=[ClassifierOutputTarget(0)])[0]
    return show_cam_on_image(rgb_uint8.astype(np.float32) / 255.0, g, use_rgb=True)
