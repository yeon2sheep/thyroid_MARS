import pandas as pd
from pathlib import Path
import numpy as np


# =========================
# 1. File paths
# =========================

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_DIR / "data" / "gwas" / "GCST90627762.tsv"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "GCST90627762_chr22.tsv"


# =========================
# 2. Load GWAS summary
# =========================

print("Loading GWAS summary statistics...")

df = pd.read_csv(
    INPUT_FILE,
    sep="\t"
)

print(f"Total variants: {len(df):,}")


# =========================
# 3. Select chromosome 22
# =========================

df = df[df["chromosome"].astype(str) == "22"].copy()

print(f"Chr22 variants: {len(df):,}")


# =========================
# 4. Calculate Z-score
# =========================

df["z"] = df["beta"] / df["standard_error"]


# =========================
# 5. Keep required columns
# =========================

columns = [
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
    "z"
]

df = df[columns]


# =========================
# 6. Basic QC
# =========================

# Remove missing values required for downstream analysis
df = df.dropna(
    subset=[
        "chromosome",
        "base_pair_location",
        "effect_allele",
        "other_allele",
        "beta",
        "standard_error",
        "p_value"
    ]
)

# Remove variants with invalid standard errors
df = df[df["standard_error"] > 0]

# Keep finite Z-scores
df = df[np.isfinite(df["z"])]


# =========================
# 7. Identify SNPs
# =========================

def is_snp(allele):
    return len(str(allele)) == 1 and str(allele).upper() in {"A", "C", "G", "T"}


df["is_snp"] = (
    df["effect_allele"].apply(is_snp)
    & df["other_allele"].apply(is_snp)
)

print(f"SNP variants: {df['is_snp'].sum():,}")
print(f"Non-SNP variants: {(~df['is_snp']).sum():,}")


# =========================
# 8. Save
# =========================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

print()
print("Preparation complete.")
print(f"Output: {OUTPUT_FILE}")
print(f"Final variants: {len(df):,}")