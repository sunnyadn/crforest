"""Build + stage the SEER breast cancer competing-risks cohort.

Source: SEER Research Data, 17 Registries, Nov 2025 Sub (2000-2023). User
must have their own SEER access (https://seerdataaccess.cancer.gov/) and
have exported a Case Listing session matching the spec in
validation/comparisons/SEER_README.md.

This script reads the raw export CSV, applies cohort filters (year
2010-2015 for homogeneous staging, drop unknown-COD deaths, drop
non-complete-dates survival), median-imputes the three numeric features
with missingness (nodes_pos, nodes_exam, cs_tumor_size), encodes
categoricals as ordinal codes, and writes:

  /tmp/seer_breast_clean.parquet   (cohort, columns x0..x16/time/status)
  /tmp/seer_breast_train_idx.txt   (80% indices)
  /tmp/seer_breast_test_idx.txt    (20% indices)

Output is consumed by validation/comparisons/seer_path_b.py.

CLI:
  python validation/gen_seer_breast.py [--src PATH] [--subsample N]

--subsample N  Random-subsample the cohort to N rows before splitting,
               for memory-constrained boxes where rfSRC OOMs at full N
               (rfSRC at n=238k, p=17 needs ~55GB). Default: full cohort.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SRC = Path.home() / "data/seer/export.csv"
DST_PARQUET = Path("/tmp/seer_breast_clean.parquet")
DST_TRAIN = Path("/tmp/seer_breast_train_idx.txt")
DST_TEST = Path("/tmp/seer_breast_test_idx.txt")
SEED = 20260430


def parse_age_band(s):
    """'50-54 years' -> 52; '90+ years' -> 92; '<1 year' -> 0.5."""
    lo = s.str.extract(r"(\d+)").iloc[:, 0].astype(float)
    hi = s.str.extract(r"-(\d+)").iloc[:, 0].astype(float)
    mid = (lo + hi) / 2
    mid = mid.fillna(lo)
    mid[s.str.startswith("<1", na=False)] = 0.5
    return mid


def parse_node_count(s):
    """SEER node codes: 0-90 valid, 95-99 special meanings -> NaN."""
    n = pd.to_numeric(s, errors="coerce")
    return n.where((n >= 0) & (n <= 90))


def parse_cs_tumor_size(s):
    """CS tumor size: 0-988 mm valid, 989+ special codes -> NaN."""
    n = pd.to_numeric(s, errors="coerce")
    return n.where((n >= 0) & (n <= 988))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument(
        "--subsample", type=int, default=None, help="random-subsample to this N before splitting"
    )
    args = parser.parse_args()

    print(f"reading {args.src} ({args.src.stat().st_size / 1e6:.1f} MB)...")
    df = pd.read_csv(args.src, dtype=str, na_values=["Blank(s)"], keep_default_na=False)
    df = df.replace({"Blank(s)": pd.NA})
    print(f"raw shape: {df.shape}")

    df["_year"] = pd.to_numeric(df["Year of diagnosis"], errors="coerce")
    df = df[(df["_year"] >= 2010) & (df["_year"] <= 2015)].copy()
    print(f"after year 2010-2015: {df.shape}")

    drop_unk = df["SEER cause-specific death classification"].eq("Dead (missing/unknown COD)")
    df = df[~drop_unk].copy()
    print(f"after drop unknown-COD: {df.shape}")

    sm_ok = df["Survival months flag"].str.startswith("Complete dates")
    df = df[sm_ok].copy()
    print(f"after survival-quality filter: {df.shape}")

    cs = df["SEER cause-specific death classification"].eq("Dead (attributable to this cancer dx)")
    oc = df["SEER other cause of death classification"].eq(
        "Dead (attributable to causes other than this cancer dx)"
    )
    status = np.where(cs, 1, np.where(oc, 2, 0)).astype(np.int64)
    time = pd.to_numeric(df["Survival months"], errors="coerce").astype(np.float64).to_numpy()

    cols_numeric = {
        "age_mid": parse_age_band(df["Age recode with <1 year olds and 90+"]),
        "year_dx": df["_year"].astype(np.float64),
        "nodes_pos": parse_node_count(df["Regional nodes positive (1988+)"]),
        "nodes_exam": parse_node_count(df["Regional nodes examined (1988+)"]),
        "cs_tumor_size": parse_cs_tumor_size(df["CS tumor size (2004-2015)"]),
    }
    cols_categorical = [
        "Sex",
        "Race recode (W, B, AI, API)",
        "Marital status at diagnosis",
        "Summary stage 2000 (1998-2017)",
        "Derived AJCC Stage Group, 7th ed (2010-2015)",
        "Histology recode - broad groupings",
        "ER Status Recode Breast Cancer (1990+)",
        "PR Status Recode Breast Cancer (1990+)",
        "Derived HER2 Recode (2010+)",
        "RX Summ--Surg Prim Site (1998-2022)",
        "Radiation recode",
        "Chemotherapy recode (yes, no/unk)",
    ]

    out = {}
    feature_names = []
    i = 0
    for name, s in cols_numeric.items():
        out[f"x{i}"] = s.astype(np.float64).to_numpy()
        feature_names.append((f"x{i}", name))
        i += 1
    for col in cols_categorical:
        codes = pd.Categorical(df[col]).codes.astype(np.float64)
        codes[codes < 0] = np.nan
        out[f"x{i}"] = codes
        feature_names.append((f"x{i}", col))
        i += 1

    out_df = pd.DataFrame(out)
    feat_cols = [c for c in out_df.columns if c.startswith("x")]

    # Median impute (only nodes_pos / nodes_exam / cs_tumor_size have NaN here).
    for c in feat_cols:
        if out_df[c].isna().any():
            out_df[c] = out_df[c].fillna(out_df[c].median())

    out_df["time"] = time
    out_df["status"] = status

    if args.subsample is not None and args.subsample < len(out_df):
        rng = np.random.default_rng(SEED)
        keep = rng.choice(len(out_df), size=args.subsample, replace=False)
        out_df = out_df.iloc[keep].reset_index(drop=True)
        print(f"subsampled to {args.subsample}: {out_df.shape}")

    out_df.to_parquet(DST_PARQUET)
    print("\nfeature mapping:")
    for x, n in feature_names:
        print(f"  {x:5s}  {n}")
    print(
        f"\nwrote {DST_PARQUET} ({DST_PARQUET.stat().st_size / 1e6:.1f} MB), shape={out_df.shape}"
    )
    print(f"status dist: {pd.Series(out_df['status']).value_counts().to_dict()}")

    rng = np.random.default_rng(SEED + 1)
    N = len(out_df)
    perm = rng.permutation(N)
    n_train = int(N * 0.8)
    np.savetxt(DST_TRAIN, perm[:n_train], fmt="%d")
    np.savetxt(DST_TEST, perm[n_train:], fmt="%d")
    print(f"wrote {DST_TRAIN} ({n_train}) + {DST_TEST} ({N - n_train})")


if __name__ == "__main__":
    main()
