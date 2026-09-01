import pandas as pd
from pathlib import Path


# =========================
# Paths
# =========================

PROJECT_DIR = Path(__file__).resolve().parents[1]

GWAS_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "GCST90627762_chr22_harmonized.tsv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "bins"
)

OUTPUT_FILE = OUTPUT_DIR / "chr22_test_bins.tsv"


# =========================
# Parameters
# =========================

BIN_SIZE = 500_000       # 500 kb
MIN_SNPS = 10
MAX_SNPS = 1000

# Pipeline test에서는 처음 20개 bin만 사용
MAX_TEST_BINS = 20


# =========================
# Load GWAS
# =========================

print("Loading harmonized GWAS...")

df = pd.read_csv(GWAS_FILE, sep="\t")

print(f"Input SNPs: {len(df):,}")


# =========================
# Basic checks
# =========================

required_columns = [
    "chromosome",
    "base_pair_location",
    "effect_allele",
    "other_allele",
    "z",
    "variant_key",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# =========================
# Sort variants
# =========================

df = df.sort_values(
    "base_pair_location"
).reset_index(drop=True)


# =========================
# Create fixed-size bins
# =========================

df["bin_start"] = (
    (df["base_pair_location"] - 1)
    // BIN_SIZE
) * BIN_SIZE + 1

df["bin_end"] = df["bin_start"] + BIN_SIZE - 1


df["bin_id"] = (
    "chr"
    + df["chromosome"].astype(str)
    + "_"
    + df["bin_start"].astype(str)
    + "_"
    + df["bin_end"].astype(str)
)


# =========================
# Summarize bins
# =========================

bins = (
    df.groupby("bin_id")
    .agg(
        chromosome=("chromosome", "first"),
        start=("bin_start", "first"),
        end=("bin_end", "first"),
        n_snp=("variant_key", "count"),
    )
    .reset_index()
)


# =========================
# Filter by SNP count
# =========================

bins = bins[
    (bins["n_snp"] >= MIN_SNPS)
    & (bins["n_snp"] <= MAX_SNPS)
].copy()


# =========================
# Keep first test bins
# =========================

bins = bins.head(MAX_TEST_BINS).copy()


# =========================
# Create output directory
# =========================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# Save bin summary
# =========================

bins.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# =========================
# Save SNP membership
# =========================

membership = df[
    df["bin_id"].isin(bins["bin_id"])
][
    [
        "bin_id",
        "chromosome",
        "base_pair_location",
        "effect_allele",
        "other_allele",
        "variant_key",
    ]
].copy()


MEMBERSHIP_FILE = (
    OUTPUT_DIR
    / "chr22_test_bin_membership.tsv"
)

membership.to_csv(
    MEMBERSHIP_FILE,
    sep="\t",
    index=False
)


# =========================
# Report
# =========================

print()
print("Bin creation complete.")
print(f"Number of test bins: {len(bins)}")
print(f"Total SNPs in test bins: {len(membership):,}")

print()
print("Bin summary:")
print(bins.to_string(index=False))

print()
print(f"Bin summary: {OUTPUT_FILE}")
print(f"SNP membership: {MEMBERSHIP_FILE}")