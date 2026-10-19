"""Upgrade pipeline: RETFound + higher resolution + richer embeddings + vessel enhancement.

Runs top to bottom on Kaggle (GPU, internet on, ODIR-5K attached, HF_TOKEN secret attached):
  1. rebuild the exact patient-level split (seed 42, same code as make_split.py)
  2. extract every candidate feature in ONE pass over the images
  3. compare 36 variants by 5-fold patient-grouped CV on train + val only
  4. refit the winner, choose its threshold out-of-fold, evaluate on test ONCE
  5. paired patient-level bootstrap vs the original models A and B
  6. fold scaler + logistic regression into one PyTorch model, Grad-CAM figure
  7. write (and optionally push) the Hugging Face Space
"""
import glob
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import model as M

T0 = time.time()
OUT = os.environ.get("OUT", "/kaggle/working")
HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = int(os.environ.get("QUICK", "0"))          # >0: only this many patients (smoke test)
FAKE = os.environ.get("FAKE", "0") == "1"           # tiny random backbones (offline smoke test)
PUSH = os.environ.get("PUSH_SPACE", "1") == "1"
SPACE_NAME = os.environ.get("SPACE_NAME", "retina-hypertension-demo")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "upgrade_log.txt"), "a")


def log(*a):
    msg = f"[{(time.time() - T0) / 60:6.1f} min] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


def hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:  # noqa: BLE001
        log("No HF_TOKEN secret available:", type(e).__name__)
        return None


# =============================================================== 1. data + split
root = os.environ.get("DATA_ROOT", "/kaggle/input")
xlsx = sorted(glob.glob(os.path.join(root, "**", "data.xlsx"), recursive=True))
assert xlsx, f"data.xlsx not found under {root}"
LABEL_PATH = xlsx[0]
IMAGE_DIR = os.path.join(os.path.dirname(LABEL_PATH), "Training Images")
assert os.path.isdir(IMAGE_DIR), IMAGE_DIR
log("labels:", LABEL_PATH)
log("images:", IMAGE_DIR, "| device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

QUALITY_TERMS = ["low image quality", "lens dust", "optic disk photographically invisible",
                 "image offset", "anterior segment image", "no fundus image"]


def norm(c):
    return str(c).strip().lower().replace("-", "_").replace(" ", "_")


df = pd.read_excel(LABEL_PATH)
df.columns = [norm(c) for c in df.columns]
df = df.drop_duplicates("id").reset_index(drop=True)
if QUICK:
    df = df.iloc[:QUICK].reset_index(drop=True)

strata = df["h"].astype(int).astype(str) + "_" + df["d"].astype(int).astype(str)
if strata.value_counts().min() < 7:
    strata = df["h"].astype(int).astype(str)
ids = df["id"].to_numpy()
train_ids, rest_ids, _, rest_strata = train_test_split(
    ids, strata.to_numpy(), test_size=0.30, stratify=strata, random_state=42)
val_ids, test_ids = train_test_split(rest_ids, test_size=0.50, stratify=rest_strata, random_state=42)
split = pd.Series("train", index=pd.Index(ids, name="patient_id"))
split.loc[val_ids] = "val"
split.loc[test_ids] = "test"

rows = []
for eye in ("left", "right"):
    rows.append(pd.DataFrame({
        "patient_id": df["id"], "eye": eye, "filename": df[f"{eye}_fundus"],
        "age": df["patient_age"], "sex": df["patient_sex"],
        "H_patient": df["h"].astype(int), "D_patient": df["d"].astype(int),
        "keywords": df[f"{eye}_diagnostic_keywords"].fillna("").astype(str)}))
meta = pd.concat(rows, ignore_index=True)
kw = meta["keywords"].str.lower()
meta["H_eye"] = kw.str.contains("hypertensi").astype(int)
meta["quality_flag"] = kw.apply(lambda s: int(any(t in s for t in QUALITY_TERMS)))
meta["split"] = meta["patient_id"].map(split)
meta = meta.sort_values(["patient_id", "eye"]).reset_index(drop=True)
pat = meta.drop_duplicates("patient_id")
key = "|".join(f"{p}:{s}" for p, s in sorted(zip(pat["patient_id"].astype(str), pat["split"])))
log("split fingerprint:", hashlib.md5(key.encode()).hexdigest())
assert (meta.groupby("patient_id")["split"].nunique() == 1).all()
exists = meta["filename"].apply(lambda f: os.path.exists(os.path.join(IMAGE_DIR, f)))
meta = meta[exists].reset_index(drop=True)
meta["sex_num"] = (meta["sex"].astype(str).str.strip().str.lower() == "male").astype(int)
log("images:", len(meta), "| positives per split:", meta.groupby("split")["H_eye"].sum().to_dict())
meta.to_csv(os.path.join(OUT, "meta_v2.csv"), index=False)

y = meta["H_eye"].to_numpy()
spl = meta["split"].to_numpy()
groups = meta["patient_id"].to_numpy()
tr, va, te = spl == "train", spl == "val", spl == "test"
dev = tr | va

# =============================================================== 2. features, one pass
FEAT_FILE = os.path.join(OUT, "features_all.npz")
BACKBONES = ["d224", "d448", "r224", "r448"]


def open_img(path):
    img = Image.open(path)
    img.draft("RGB", (1024, 1024))                     # fast JPEG decode, still >= 1024 px
    return img


class FundusDS(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        b = M.base_image(open_img(os.path.join(IMAGE_DIR, self.files[i])))
        return M.to_tensor(b), M.to_tensor(M.clahe(b)), i


def fake_backbones():
    import timm
    mk = {"d": lambda s: timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=0, img_size=s),
          "r": lambda s: timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0, img_size=s)}
    torch.manual_seed(0)
    return {k: mk[k[0]](M.RES[k]).eval() for k in BACKBONES}


TOKEN = hf_token()
if FAKE:
    M.load_backbones = lambda keys, token=None: {k: v for k, v in fake_backbones().items() if k in keys}

if os.path.exists(FEAT_FILE):
    log("reusing", FEAT_FILE)
    z = np.load(FEAT_FILE)
    feats = {k: z[k].astype(np.float32) for k in z.files}
else:
    models = M.load_backbones(BACKBONES, TOKEN)
    for m in models.values():
        m.to(DEVICE)
        for p in m.parameters():
            p.requires_grad = False
    log("backbones loaded:", {k: sum(p.numel() for p in m.parameters()) // 10**6 for k, m in models.items()}, "M params")
    loader = DataLoader(FundusDS(meta["filename"].tolist()), batch_size=32, shuffle=False,
                        num_workers=4 if DEVICE == "cuda" else 0, pin_memory=DEVICE == "cuda")
    feats, seen = {}, []
    use_amp = DEVICE == "cuda"
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        for k, (xp, xc, idx) in enumerate(loader):
            for prep, X in (("p", xp), ("c", xc)):
                X = X.to(DEVICE, non_blocking=True)
                for fl in ("o", "f"):
                    Xi = torch.flip(X, dims=[3]) if fl == "f" else X
                    tok = M.token_features(models, Xi, BACKBONES)
                    for name in M.FEATURES:
                        v = M.pooled(models, tok, name).float().cpu().numpy()
                        feats.setdefault(f"{name}_{prep}_{fl}", []).append(v)
            seen.append(idx.numpy())
            if k % 20 == 0:
                log(f"batch {k + 1}/{len(loader)}")
    assert (np.concatenate(seen) == np.arange(len(meta))).all(), "row order broken"
    feats = {k: np.concatenate(v) for k, v in feats.items()}
    for k, v in feats.items():
        assert np.isfinite(v).all(), f"non-finite values in {k}"
    np.savez(FEAT_FILE, **{k: v.astype(np.float16) for k, v in feats.items()})
    del models
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    log("features saved:", FEAT_FILE, {k: v.shape[1] for k, v in feats.items() if k.endswith("_p_o")})

# =============================================================== 3. variants + CV on dev only
SETS = {
    "DINOv2-224 CLS": ["d224_cls"],
    "DINOv2-448 CLS": ["d448_cls"],
    "DINOv2-448 CLS+mean": ["d448_cls", "d448_mean"],
    "RETFound-224 mean": ["r224_mean"],
    "RETFound-224 CLS+mean": ["r224_cls", "r224_mean"],
    "RETFound-448 mean": ["r448_mean"],
    "RETFound-448 CLS+mean": ["r448_cls", "r448_mean"],
    "DINOv2-448 CLS+mean + RETFound-224 mean": ["d448_cls", "d448_mean", "r224_mean"],
    "DINOv2-448 CLS+mean + RETFound-448 mean": ["d448_cls", "d448_mean", "r448_mean"],
}
VARIANTS = []
for set_name, names in SETS.items():
    for prep in ("p", "c"):
        for tta in (False, True):
            VARIANTS.append({"set": set_name, "features": names, "clahe": prep == "c", "tta": tta})


def build_X(v):
    prep = "c" if v["clahe"] else "p"
    cols = []
    for n in v["features"]:
        a = feats[f"{n}_{prep}_o"]
        cols.append((a + feats[f"{n}_{prep}_f"]) / 2 if v["tta"] else a)
    return np.hstack(cols).astype(np.float32)


def vname(v):
    return f"{v['set']} | {'CLAHE' if v['clahe'] else 'plain'} | {'flip-TTA' if v['tta'] else 'no TTA'}"


def cost(v):
    n_models = len({M.FEATURES[n][0] for n in v["features"]})
    passes = n_models * (2 if v["tta"] else 1)
    heavy = sum(4 if b.startswith("r448") else 1 for b in {M.FEATURES[n][0] for n in v["features"]})
    return (passes * heavy, build_X(v).shape[1], int(v["clahe"]))


cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
y_dev, g_dev = y[dev], groups[dev]
FOLDS = list(cv.split(np.zeros(dev.sum()), y_dev, g_dev))
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]


def fit_lr(X, yy, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(sc.transform(X), yy)
    return sc, clf


def cv_one(X_dev, C):
    oof = np.zeros(len(y_dev))
    aucs = []
    for trn, tst in FOLDS:
        sc, clf = fit_lr(X_dev[trn], y_dev[trn], C)
        oof[tst] = clf.decision_function(sc.transform(X_dev[tst]))
        aucs.append(roc_auc_score(y_dev[tst], oof[tst]))
    return float(np.mean(aucs)), float(np.std(aucs)), oof


from joblib import Parallel, delayed  # noqa: E402

CV_FILE = os.path.join(OUT, "cv_results.csv")
log(f"CV: {len(VARIANTS)} variants x {len(C_GRID)} C values x 5 folds on {dev.sum()} dev images "
    f"({y_dev.sum()} positive)")
rows = []
for vi, v in enumerate(VARIANTS):
    X_dev = build_X(v)[dev]
    res = Parallel(n_jobs=-1)(delayed(cv_one)(X_dev, C) for C in C_GRID)
    for C, (mu, sd, _) in zip(C_GRID, res):
        rows.append({"variant": vname(v), "vi": vi, "set": v["set"], "clahe": v["clahe"], "tta": v["tta"],
                     "dims": X_dev.shape[1], "C": C, "cv_auroc": mu, "cv_std": sd})
    best = max(res, key=lambda r: r[0])
    log(f"  {vi + 1:2d}/{len(VARIANTS)} {vname(v):70s} best CV AUROC {best[0]:.3f} ± {best[1]:.3f}")
cvdf = pd.DataFrame(rows)
cvdf.to_csv(CV_FILE, index=False)

best_per = cvdf.loc[cvdf.groupby("vi")["cv_auroc"].idxmax()].sort_values("cv_auroc", ascending=False)
top = best_per["cv_auroc"].max()
cands = best_per[best_per["cv_auroc"] >= top - 0.01].copy()
cands["cost"] = cands["vi"].apply(lambda i: cost(VARIANTS[i]))
winner_row = cands.sort_values("cost").iloc[0]
W = VARIANTS[int(winner_row["vi"])]
C_W = float(winner_row["C"])
log("Selection rule: highest mean CV AUROC; within 0.01 of the best, the cheapest variant wins.")
log("Top variants:\n" + best_per.head(12)[["variant", "dims", "C", "cv_auroc", "cv_std"]].to_string(index=False))
log(f"WINNER: {vname(W)}  C={C_W}  CV AUROC {winner_row['cv_auroc']:.3f} ± {winner_row['cv_std']:.3f}")


def factor_effect(col, a, b):
    """Mean CV gain of b over a, matched on everything else."""
    p = best_per.pivot_table(index=[c for c in ["set", "clahe", "tta"] if c != col], columns=col, values="cv_auroc")
    return float((p[b] - p[a]).mean())


abl = {
    "resolution 224 -> 448 (DINOv2 CLS)": float(best_per[best_per.set == "DINOv2-448 CLS"].cv_auroc.mean()
                                                 - best_per[best_per.set == "DINOv2-224 CLS"].cv_auroc.mean()),
    "resolution 224 -> 448 (RETFound mean)": float(best_per[best_per.set == "RETFound-448 mean"].cv_auroc.mean()
                                                   - best_per[best_per.set == "RETFound-224 mean"].cv_auroc.mean()),
    "richer: CLS -> CLS+mean (DINOv2-448)": float(best_per[best_per.set == "DINOv2-448 CLS+mean"].cv_auroc.mean()
                                                  - best_per[best_per.set == "DINOv2-448 CLS"].cv_auroc.mean()),
    "backbone: best DINOv2 -> best RETFound": float(best_per[best_per.set.str.startswith("RETFound")].cv_auroc.max()
                                                    - best_per[best_per.set.str.startswith("DINOv2") & ~best_per.set.str.contains("RETFound")].cv_auroc.max()),
    "CLAHE (avg over matched variants)": factor_effect("clahe", False, True),
    "flip TTA (avg over matched variants)": factor_effect("tta", False, True),
}
log("Effect of each improvement on CV AUROC:\n" + "\n".join(f"  {k:45s} {v:+.3f}" for k, v in abl.items()))
json.dump({"winner": vname(W), "C": C_W, "cv_auroc": float(winner_row["cv_auroc"]),
           "cv_std": float(winner_row["cv_std"]), "ablation": abl},
          open(os.path.join(OUT, "selection.json"), "w"), indent=2)

# =============================================================== 4. original A and B (same recipe as before)
demo = meta[["age", "sex_num"]].to_numpy(dtype=float)
scA, clfA = StandardScaler().fit(demo[tr]), None
clfA = LogisticRegression(class_weight="balanced", max_iter=1000).fit(scA.transform(demo[tr]), y[tr])
V0 = feats["d224_cls_p_o"]
val_aucs = []
for C in [0.0001, 0.001, 0.01, 0.1, 1]:
    sc, clf = StandardScaler().fit(V0[tr]), None
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(sc.transform(V0[tr]), y[tr])
    val_aucs.append((roc_auc_score(y[va], clf.predict_proba(sc.transform(V0[va]))[:, 1]), C, sc, clf))
aucB, C_B, scB, clfB = max(val_aucs, key=lambda t: t[0])
scores = {
    "A": {s: clfA.predict_proba(scA.transform(demo[m]))[:, 1] for s, m in (("val", va), ("test", te))},
    "B": {s: clfB.predict_proba(scB.transform(V0[m]))[:, 1] for s, m in (("val", va), ("test", te))},
}
log(f"Original recipe reproduced: A val AUROC {roc_auc_score(y[va], scores['A']['val']):.3f} (was 0.602), "
    f"B val AUROC {aucB:.3f} at C={C_B} (was 0.795)")


def threshold_at(yy, s, target=0.85):
    fpr, tpr, thr = roc_curve(yy, s)
    i = int(np.argmax(tpr >= target))
    return float(thr[i])


thr = {n: threshold_at(y[va], scores[n]["val"]) for n in ("A", "B")}

# =============================================================== 5. winner: refit on dev, OOF threshold, ONE test look
X_W = build_X(W)
scW, clfW = fit_lr(X_W[dev], y[dev], C_W)
_, _, oof = cv_one(X_W[dev], C_W)
thr["B2"] = threshold_at(y_dev, oof)
scores["B2"] = {"test": clfW.decision_function(scW.transform(X_W[te]))}
yt = y[te]
res = []
for n in ("A", "B", "B2"):
    s = scores[n]["test"]
    pred = (s >= thr[n]).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    res.append({"model": n, "test_auroc": roc_auc_score(yt, s), "test_ap": average_precision_score(yt, s),
                "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp), "tp": tp, "fn": fn, "fp": fp})
test_df = pd.DataFrame(res).set_index("model")
test_df.to_csv(os.path.join(OUT, "test_results_v2.csv"))
log(f"TEST (second evaluation; positives {int(yt.sum())}, chance AP {yt.mean():.3f}):\n" + test_df.round(3).to_string())

# paired patient-level bootstrap
tmeta = meta.loc[te].reset_index(drop=True)
pids = tmeta["patient_id"].unique()
rows_of = tmeta.groupby("patient_id").indices
rng = np.random.default_rng(0)
boot = {k: [] for k in ["A", "B", "B2", "B2-B", "B2-A", "B-A"]}
for _ in range(1000):
    idx = np.concatenate([rows_of[p] for p in rng.choice(pids, size=len(pids), replace=True)])
    yb = yt[idx]
    if yb.min() == yb.max():
        continue
    a = {m: roc_auc_score(yb, scores[m]["test"][idx]) for m in ("A", "B", "B2")}
    for m in a:
        boot[m].append(a[m])
    boot["B2-B"].append(a["B2"] - a["B"])
    boot["B2-A"].append(a["B2"] - a["A"])
    boot["B-A"].append(a["B"] - a["A"])
ci = pd.DataFrame([{"metric": k, "mean": np.mean(v), "ci_low": np.percentile(v, 2.5),
                    "ci_high": np.percentile(v, 97.5)} for k, v in boot.items()]).set_index("metric")
ci.to_csv(os.path.join(OUT, "bootstrap_ci_v2.csv"))
log("Bootstrap (1000 rounds, patients resampled):\n" + ci.round(3).to_string())

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    lab = {"A": "A  age + sex", "B": "B  DINOv2-224 (original)", "B2": "B2 " + W["set"]}
    for n in ("A", "B", "B2"):
        fpr, tpr, _ = roc_curve(yt, scores[n]["test"])
        ax[0].plot(fpr, tpr, label=f"{lab[n]}  AUROC {roc_auc_score(yt, scores[n]['test']):.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0].set(xlabel="False positive rate", ylabel="Sensitivity", title="Test ROC")
    ax[0].legend(fontsize=8)
    pt, pp = calibration_curve(yt, 1 / (1 + np.exp(-scores["B2"]["test"])), n_bins=5, strategy="quantile")
    ax[1].plot(pp, pt, "o-", label="B2")
    ax[1].plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax[1].set(xlabel="Mean predicted probability", ylabel="Observed fraction positive", title="Calibration (test, B2)")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "test_roc_calibration_v2.png"), dpi=150)
    plt.close()
except Exception as e:  # noqa: BLE001
    log("plot failed:", e)

# =============================================================== 6. one PyTorch model + match check + Grad-CAM
w_p, b_p = M.fold_head(scW, clfW)
need = sorted({M.FEATURES[n][0] for n in W["features"]})
bb = fake_backbones() if FAKE else M.load_backbones(need, TOKEN)
rm = M.RetinaModel({k: bb[k] for k in need}, W["features"], w_p, b_p, W["tta"]).to(DEVICE).eval()
test_idx = np.where(te)[0]
check = test_idx[:16]
xs = []
for i in check:
    _, x = M.prepare(open_img(os.path.join(IMAGE_DIR, meta.loc[i, "filename"])), W["clahe"])
    xs.append(x)
xs = torch.stack(xs).to(DEVICE)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
    f = rm.features(xs)
    if W["tta"]:
        f = (f + rm.features(torch.flip(xs, dims=[3]))) / 2
with torch.no_grad():
    logit_model = (f.float() @ rm.head.weight.float().T + rm.head.bias.float()).squeeze(1).cpu().numpy()
logit_sk = clfW.decision_function(scW.transform(X_W[check]))
diff = float(np.abs(logit_model - logit_sk).max())
log(f"Match check (16 test images, same fp16 path): max |logit diff| = {diff:.4f} (logit spread {logit_sk.std():.2f})")
with torch.no_grad():
    rm32 = rm.float()
    full_logit = rm32(xs.float()).squeeze(1).cpu().numpy()
log(f"Full fp32 model vs sklearn: max |diff| = {np.abs(full_logit - logit_sk).max():.4f}")

try:
    import matplotlib.pyplot as plt
    s_te = scores["B2"]["test"]
    pos, neg = test_idx[yt == 1], test_idx[yt == 0]
    sp, sn = s_te[yt == 1], s_te[yt == 0]
    order_p = np.argsort(-sp)
    picks = [(pos[order_p[0]], "true positive"), (pos[order_p[1]], "true positive"),
             (pos[order_p[-1]], "missed positive"), (neg[int(np.argmax(sn))], "top false positive")]
    cam = M.make_cam(rm32)
    fig, ax = plt.subplots(2, 4, figsize=(16, 8.4))
    for j, (i, title) in enumerate(picks):
        rgb, x = M.prepare(open_img(os.path.join(IMAGE_DIR, meta.loc[i, "filename"])), W["clahe"])
        ov = M.cam_overlay(cam, x.to(DEVICE), rgb)
        rank = (s_te < s_te[test_idx.tolist().index(i)]).mean() * 100
        ax[0, j].imshow(rgb)
        ax[0, j].set_title(f"{title}\nscore above {rank:.0f}% of test images", fontsize=10)
        ax[1, j].imshow(ov)
        for a in ax[:, j]:
            a.axis("off")
    plt.suptitle(f"Grad-CAM, {W['set']} ({'CLAHE' if W['clahe'] else 'plain'})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "gradcam_examples.png"), dpi=120)
    plt.close()
    log("Grad-CAM figure saved")
except Exception as e:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    log("Grad-CAM failed:", e)

# =============================================================== 7. Hugging Face Space
SPACE = os.path.join(OUT, "space")
os.makedirs(SPACE, exist_ok=True)
for f_ in ("model.py", "app.py"):
    shutil.copy(os.path.join(HERE, f_), os.path.join(SPACE, f_))
t_sens = float(test_df.loc["B2", "sensitivity"])
t_spec = float(test_df.loc["B2", "specificity"])
np.savez(os.path.join(SPACE, "head.npz"), w=w_p, b=np.float32(b_p), threshold=np.float32(thr["B2"]),
         test_logits=scores["B2"]["test"].astype(np.float32),
         spec=json.dumps({"features": W["features"], "clahe": W["clahe"], "tta": W["tta"], "variant": vname(W),
                          "test_auroc": float(test_df.loc["B2", "test_auroc"]),
                          "auroc_ci": [float(ci.loc["B2", "ci_low"]), float(ci.loc["B2", "ci_high"])],
                          "sensitivity": t_sens, "specificity": t_spec}))
import timm  # noqa: E402

uses_retfound = any(k.startswith("r") for k in need)
open(os.path.join(SPACE, "requirements.txt"), "w").write(
    f"torch\ntorchvision\ntimm=={timm.__version__}\ngrad-cam\nopencv-python-headless\nhuggingface_hub\nsafetensors\nnumpy\npillow\n")
log("Space files written to", SPACE, os.listdir(SPACE))

if PUSH and TOKEN and not FAKE:
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=TOKEN)
        user = api.whoami()["name"]
        repo = f"{user}/{SPACE_NAME}"
        api.create_repo(repo, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
        sdk_version = None
        try:
            old = open(hf_hub_download(repo, "README.md", repo_type="space", token=TOKEN)).read()
            sdk_version = next((l.split(":", 1)[1].strip() for l in old.splitlines() if l.startswith("sdk_version")), None)
        except Exception:  # noqa: BLE001
            pass
        card = ["---", "title: Retinal Hypertension Signs", "emoji: 👁️", "colorFrom: indigo", "colorTo: red",
                "sdk: gradio"] + ([f"sdk_version: {sdk_version}"] if sdk_version else []) + [
                "app_file: app.py", "pinned: false"] + (["license: cc-by-nc-4.0"] if uses_retfound else []) + [
                "---", "",
                "# Retinal hypertension signs: research demo",
                "",
                "Upload a colour fundus photo. The model scores it for signs of hypertensive retinopathy, "
                "shows where it looked (Grad-CAM) and ranks the score against a held-out test set.",
                "",
                f"Model: {vname(W)} with a logistic-regression head, trained on ODIR-5K. "
                f"Test AUROC {test_df.loc['B2', 'test_auroc']:.2f} "
                f"(95% CI {ci.loc['B2', 'ci_low']:.2f}–{ci.loc['B2', 'ci_high']:.2f}).",
                "",
                "**Not a medical device.** Single dataset, small number of positive cases, no external validation.",
                ] + (["", "RETFound weights (CC BY-NC 4.0) are gated: add a read token as the Space secret `HF_TOKEN`."]
                     if uses_retfound else [])
        open(os.path.join(SPACE, "README.md"), "w").write("\n".join(card) + "\n")
        api.upload_folder(folder_path=SPACE, repo_id=repo, repo_type="space",
                          commit_message="Upgrade: model, Grad-CAM demo")
        log(f"SPACE PUSHED: https://huggingface.co/spaces/{repo}")
    except Exception as e:  # noqa: BLE001
        log("Space push failed (do it by hand from /kaggle/working/space):", type(e).__name__, str(e)[:300])

log("DONE")
