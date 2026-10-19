#!/usr/bin/env python3
"""
Patient-level, stratified train / val / test split for ODIR-5K.

Usage
-----
  python make_split.py data.xlsx                 # writes splits.csv and prints a report
  python make_split.py data.xlsx --seed 42 --out splits.csv
  python make_split.py --verify splits.csv       # re-checks an existing split file

Input
-----
The ODIR-5K label sheet (data.xlsx on Kaggle): one row per PATIENT, with columns
  ID, Patient Age, Patient Sex, Left-Fundus, Right-Fundus,
  Left-Diagnostic Keywords, Right-Diagnostic Keywords, N, D, G, C, A, H, M, O

Output (splits.csv)
-------------------
One row per IMAGE (two per patient):
  patient_id, eye, filename, age, sex, split,
  H_patient, D_patient      patient-level labels from the sheet (apply to BOTH eyes)
  H_eye                     1 if THIS eye's keywords mention hypertensive changes
  keywords                  this eye's raw diagnostic keywords
  quality_flag              1 if the keywords say the image is unusable / poor

Why patient-level: both eyes of one person must land in the same split,
otherwise the model can recognise the person instead of the disease.
"""
import argparse
import hashlib
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED = [
    "id", "patient_age", "patient_sex", "left_fundus", "right_fundus",
    "left_diagnostic_keywords", "right_diagnostic_keywords", "h", "d",
]
QUALITY_TERMS = [
    "low image quality", "lens dust", "optic disk photographically invisible",
    "image offset", "anterior segment image", "no fundus image",
]
MIN_PER_STRATUM = 7  # below this a 70/15/15 split cannot place one in every part


def norm(col: str) -> str:
    return str(col).strip().lower().replace("-", "_").replace(" ", "_")


def load_patients(path: str) -> pd.DataFrame:
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df.columns = [norm(c) for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns {missing}.\nFound: {list(df.columns)}")
    if df["id"].duplicated().any():
        n = int(df["id"].duplicated().sum())
        print(f"NOTE: {n} repeated patient IDs (file looks image-level); keeping one row per patient.")
        df = df.drop_duplicates("id")
    return df.reset_index(drop=True)


def split_patients(df: pd.DataFrame, seed: int) -> pd.Series:
    """Return a Series patient_id -> split, 70/15/15, stratified."""
    strata = df["h"].astype(int).astype(str) + "_" + df["d"].astype(int).astype(str)
    label = "hypertension x diabetes"
    if strata.value_counts().min() < MIN_PER_STRATUM:
        strata, label = df["h"].astype(int).astype(str), "hypertension only"
    print(f"Stratifying on: {label}")

    ids = df["id"].to_numpy()
    train_ids, rest_ids, _, rest_strata = train_test_split(
        ids, strata.to_numpy(), test_size=0.30, stratify=strata, random_state=seed)
    val_ids, test_ids = train_test_split(
        rest_ids, test_size=0.50, stratify=rest_strata, random_state=seed)

    out = pd.Series("train", index=pd.Index(ids, name="patient_id"), name="split")
    out.loc[val_ids] = "val"
    out.loc[test_ids] = "test"
    return out


def to_image_rows(df: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    rows = []
    for eye in ("left", "right"):
        part = pd.DataFrame({
            "patient_id": df["id"],
            "eye": eye,
            "filename": df[f"{eye}_fundus"],
            "age": df["patient_age"],
            "sex": df["patient_sex"],
            "H_patient": df["h"].astype(int),
            "D_patient": df["d"].astype(int),
            "keywords": df[f"{eye}_diagnostic_keywords"].fillna("").astype(str),
        })
        rows.append(part)
    img = pd.concat(rows, ignore_index=True)
    kw = img["keywords"].str.lower()
    img["H_eye"] = kw.str.contains("hypertensi").astype(int)
    img["quality_flag"] = kw.apply(lambda s: int(any(t in s for t in QUALITY_TERMS)))
    img["split"] = img["patient_id"].map(split)
    cols = ["patient_id", "eye", "filename", "age", "sex", "split",
            "H_patient", "D_patient", "H_eye", "keywords", "quality_flag"]
    return img[cols].sort_values(["patient_id", "eye"]).reset_index(drop=True)


def check_and_report(img: pd.DataFrame) -> None:
    """Hard checks first (fail loudly), then a table to eyeball."""
    # 1. every image has a split
    assert img["split"].isin(["train", "val", "test"]).all(), "rows without a valid split"
    # 2. no patient appears in more than one split
    per_patient = img.groupby("patient_id")["split"].nunique()
    assert (per_patient == 1).all(), f"{int((per_patient > 1).sum())} patients span several splits"
    # 3. explicit set intersections, the check a reviewer would ask for
    sets = {s: set(g["patient_id"]) for s, g in img.groupby("split")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (sets[a] & sets[b]), f"patient overlap between {a} and {b}"
    # 4. no image file listed twice
    assert not img["filename"].duplicated().any(), "duplicate image filenames"

    pat = img.drop_duplicates("patient_id")
    rep = pd.DataFrame({
        "patients": pat.groupby("split").size(),
        "images": img.groupby("split").size(),
        "H_patients": pat.groupby("split")["H_patient"].sum(),
        "H_prev_%": (pat.groupby("split")["H_patient"].mean() * 100).round(2),
        "H_eye_images": img.groupby("split")["H_eye"].sum(),
        "D_patients": pat.groupby("split")["D_patient"].sum(),
        "D_prev_%": (pat.groupby("split")["D_patient"].mean() * 100).round(2),
        "mean_age": pat.groupby("split")["age"].mean().round(1),
        "flagged_images": img.groupby("split")["quality_flag"].sum(),
    }).reindex(["train", "val", "test"])
    print("\nAll leakage checks passed: no patient is in more than one split.\n")
    print(rep.to_string())
    print("\nSex by split (counts):")
    print(pd.crosstab(pat["split"], pat["sex"]).reindex(["train", "val", "test"]).to_string())

    both = int(((img.groupby("patient_id")["H_eye"].sum() == 1)).sum())
    print(f"\nPatients where only ONE eye has hypertensive keywords: {both}")
    print("  -> for those, H_patient = 1 on both eyes but H_eye = 1 on one. Decide which label you train on.")

    key = "|".join(f"{p}:{s}" for p, s in sorted(zip(pat["patient_id"].astype(str), pat["split"])))
    print(f"\nSplit fingerprint (md5): {hashlib.md5(key.encode()).hexdigest()}")
    print("  -> same seed + same input file must reproduce this exact value.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="?", help="ODIR-5K label sheet (data.xlsx or .csv)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="splits.csv")
    ap.add_argument("--verify", metavar="SPLITS_CSV", help="only re-check an existing splits.csv")
    args = ap.parse_args()

    if args.verify:
        check_and_report(pd.read_csv(args.verify, keep_default_na=False))
        return
    if not args.labels:
        ap.error("give the label sheet, or use --verify splits.csv")

    df = load_patients(args.labels)
    print(f"Loaded {len(df)} patients from {args.labels} (seed {args.seed})")
    img = to_image_rows(df, split_patients(df, args.seed))
    check_and_report(img)
    img.to_csv(args.out, index=False)
    print(f"\nWrote {len(img)} image rows to {args.out}")


if __name__ == "__main__":
    main()
