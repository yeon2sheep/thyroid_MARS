import pandas as pd
from pathlib import Path


# =========================
# Paths
# =========================

PROJECT_DIR = Path(__file__).resolve().parents[1]

GWAS_FILE = PROJECT_DIR / "data" / "processed" / "GCST90627762_chr22.tsv"

OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "GCST90627762_chr22_harmonized.tsv"


# =========================
# Load GWAS
# =========================

print("Loading GWAS summary statistics...")

df = pd.read_csv(GWAS_FILE, sep="\t")

print(f"Input variants: {len(df):,}")


# =========================
# Standardize chromosome
# =========================

df["chromosome"] = df["chromosome"].astype(str)
df["chromosome"] = df["chromosome"].str.replace("chr", "", regex=False)


# =========================
# Calculate Z-score
# =========================

df["z"] = df["beta"] / df["standard_error"]


# =========================
# Remove invalid statistics
# =========================

before = len(df)

df = df.dropna(subset=[
    "chromosome",
    "base_pair_location",
    "effect_allele",
    "other_allele",
    "beta",
    "standard_error",
    "p_value"
])

df = df[df["standard_error"] > 0]

after = len(df)

print(f"Removed invalid rows: {before - after:,}")


# =========================
# Keep biallelic SNPs
# =========================

def is_snp(row):
    return (
        len(row["effect_allele"]) == 1
        and len(row["other_allele"]) == 1
        and row["effect_allele"] in {"A", "C", "G", "T"}
        and row["other_allele"] in {"A", "C", "G", "T"}
    )


df = df[df.apply(is_snp, axis=1)].copy()

print(f"SNP variants: {len(df):,}")


# =========================
# Create variant key
# =========================

df["variant_key"] = (
    df["chromosome"].astype(str)
    + ":"
    + df["base_pair_location"].astype(str)
    + ":"
    + df["effect_allele"]
    + ":"
    + df["other_allele"]
)


# =========================
# Remove duplicated variants
# =========================

duplicates = df["variant_key"].duplicated().sum()

print(f"Duplicated variants: {duplicates:,}")

df = df.drop_duplicates("variant_key").copy()


# =========================
# Sort variants
# =========================

df = df.sort_values(
    ["chromosome", "base_pair_location"]
).reset_index(drop=True)


# =========================
# Save
# =========================

df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

print()
print("Harmonization preparation complete.")
print(f"Output: {OUTPUT_FILE}")
print(f"Final variants: {len(df):,}")