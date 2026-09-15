import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "gwas"
    / "GCST90627762.tsv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "GCST90627762_chr22_prepared.tsv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Load GWAS summary statistics
# ============================================================

print("=" * 70)
print("01. GWAS PREPARATION")
print("=" * 70)

print(f"Input: {INPUT_FILE}")

df = pd.read_csv(
    INPUT_FILE,
    sep="\t"
)

print(f"Total GWAS variants: {len(df):,}")


# ============================================================
# 3. Select chr22
# ============================================================

df["chromosome"] = (
    df["chromosome"]
    .astype(str)
    .str.replace("chr", "", regex=False)
)

df = df[
    df["chromosome"] == "22"
].copy()

print(f"Chr22 variants: {len(df):,}")


# ============================================================
# 4. Keep required columns
# ============================================================

required_columns = [
    "chromosome",
    "base_pair_location",
    "effect_allele",
    "other_allele",
    "beta",
    "standard_error",
    "effect_allele_frequency",
    "p_value",
    "variant_id",
    "n",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing GWAS columns: {missing_columns}"
    )

df = df[
    required_columns
].copy()


# ============================================================
# 5. Basic cleaning
# ============================================================

df["base_pair_location"] = pd.to_numeric(
    df["base_pair_location"],
    errors="coerce"
)

df["beta"] = pd.to_numeric(
    df["beta"],
    errors="coerce"
)

df["standard_error"] = pd.to_numeric(
    df["standard_error"],
    errors="coerce"
)

df["p_value"] = pd.to_numeric(
    df["p_value"],
    errors="coerce"
)

df["effect_allele"] = (
    df["effect_allele"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["other_allele"] = (
    df["other_allele"]
    .astype(str)
    .str.strip()
    .str.upper()
)


before = len(df)

df = df.dropna(
    subset=[
        "base_pair_location",
        "effect_allele",
        "other_allele",
        "beta",
        "standard_error",
        "p_value",
    ]
)

print(
    f"Removed missing required values: "
    f"{before - len(df):,}"
)


# ============================================================
# 6. Keep valid biallelic SNPs
# ============================================================

valid_bases = {
    "A",
    "C",
    "G",
    "T",
}


def is_snp(row):
    return (
        len(row["effect_allele"]) == 1
        and len(row["other_allele"]) == 1
        and row["effect_allele"] in valid_bases
        and row["other_allele"] in valid_bases
        and row["effect_allele"] != row["other_allele"]
    )


snp_mask = df.apply(
    is_snp,
    axis=1
)

print(
    f"Non-SNP / invalid allele rows removed: "
    f"{(~snp_mask).sum():,}"
)

df = df[snp_mask].copy()


# ============================================================
# 7. Remove invalid standard errors
# ============================================================

before = len(df)

df = df[
    df["standard_error"] > 0
].copy()

print(
    f"Invalid SE removed: "
    f"{before - len(df):,}"
)


# ============================================================
# 8. Calculate Z-score
# ============================================================

df["z"] = (
    df["beta"]
    / df["standard_error"]
)

invalid_z = (
    df["z"].isna()
    | ~np.isfinite(df["z"])
)

print(
    f"Invalid Z-scores removed: "
    f"{invalid_z.sum():,}"
)

df = df[
    ~invalid_z
].copy()


# ============================================================
# 9. Create variant key
# ============================================================

df["variant_key"] = (
    df["chromosome"].astype(str)
    + ":"
    + df["base_pair_location"].astype(int).astype(str)
    + ":"
    + df["effect_allele"]
    + ":"
    + df["other_allele"]
)


# ============================================================
# 10. Remove duplicate variants
# ============================================================

duplicate_count = (
    df["variant_key"]
    .duplicated()
    .sum()
)

print(
    f"Duplicate variants removed: "
    f"{duplicate_count:,}"
)

df = df[
    ~df["variant_key"].duplicated()
].copy()


# ============================================================
# 11. Sort
# ============================================================

df = df.sort_values(
    [
        "base_pair_location",
        "variant_key",
    ]
).reset_index(
    drop=True
)


# ============================================================
# 12. Save
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# 13. Summary
# ============================================================

print()
print("=" * 70)
print("GWAS PREPARATION COMPLETE")
print("=" * 70)

print(
    f"Final chr22 SNPs : {len(df):,}"
)

print(
    f"Output            : {OUTPUT_FILE}"
)

print()
print("Columns:")
print(
    df.columns.tolist()
)