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

BIN_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "bins"
    / "chr22_test_bins.tsv"
)

MEMBERSHIP_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "bins"
    / "chr22_test_bin_membership.tsv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "mars_input"
)


# =========================
# Load data
# =========================

print("Loading GWAS data...")
gwas = pd.read_csv(GWAS_FILE, sep="\t")

print("Loading bin information...")
bins = pd.read_csv(BIN_FILE, sep="\t")

print("Loading SNP membership...")
membership = pd.read_csv(MEMBERSHIP_FILE, sep="\t")


print()
print(f"GWAS SNPs: {len(gwas):,}")
print(f"Test bins: {len(bins):,}")
print(f"Membership records: {len(membership):,}")


# =========================
# Check required columns
# =========================

required_gwas = [
    "variant_key",
    "z",
]

required_membership = [
    "bin_id",
    "variant_key",
]

for column in required_gwas:
    if column not in gwas.columns:
        raise ValueError(
            f"GWAS column missing: {column}"
        )

for column in required_membership:
    if column not in membership.columns:
        raise ValueError(
            f"Membership column missing: {column}"
        )


# =========================
# Merge Z-score into membership
# =========================

print()
print("Matching SNPs with GWAS Z-scores...")

mars_data = membership.merge(
    gwas[
        [
            "variant_key",
            "z",
            "p_value",
            "beta",
            "standard_error",
            "effect_allele",
            "other_allele",
        ]
    ],
    on="variant_key",
    how="left",
    validate="one_to_one",
)


# =========================
# Check matching
# =========================

missing_z = mars_data["z"].isna().sum()

print(f"Missing Z-scores: {missing_z:,}")

if missing_z > 0:
    print("WARNING: Some SNPs do not have GWAS statistics.")
    mars_data = mars_data.dropna(subset=["z"])


# =========================
# Create output directory
# =========================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# Save complete SNP table
# =========================

ALL_SNP_FILE = (
    OUTPUT_DIR
    / "chr22_test_mars_snps.tsv"
)

mars_data.to_csv(
    ALL_SNP_FILE,
    sep="\t",
    index=False
)


# =========================
# Create one file per bin
# =========================

print()
print("Creating bin-specific MARS input files...")

created = 0

for bin_id in bins["bin_id"]:

    bin_data = mars_data[
        mars_data["bin_id"] == bin_id
    ].copy()

    if len(bin_data) == 0:
        continue

    # Make SNP order explicit.
    # This order must later be identical
    # to the genotype/LD matrix order.
    bin_data = bin_data.sort_values(
        ["chromosome", "base_pair_location"]
    ).reset_index(drop=True)

    # Add SNP index
    bin_data.insert(
        0,
        "snp_index",
        range(1, len(bin_data) + 1)
    )

    safe_bin_id = bin_id.replace(":", "_")

    output_file = (
        OUTPUT_DIR
        / f"{safe_bin_id}.tsv"
    )

    bin_data.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    created += 1

    print(
        f"  {bin_id}: "
        f"{len(bin_data):,} SNPs"
    )


# =========================
# Save summary
# =========================

summary = (
    mars_data
    .groupby("bin_id")
    .agg(
        n_snp=("variant_key", "count"),
        mean_z=("z", "mean"),
        max_abs_z=("z", lambda x: x.abs().max()),
        min_p=("p_value", "min"),
    )
    .reset_index()
)

summary = summary.merge(
    bins,
    on="bin_id",
    how="left"
)

summary_file = (
    OUTPUT_DIR
    / "chr22_test_mars_input_summary.tsv"
)

summary.to_csv(
    summary_file,
    sep="\t",
    index=False
)


# =========================
# Final report
# =========================

print()
print("MARS input preparation complete.")
print(f"Created bin files: {created}")
print(f"All SNPs: {ALL_SNP_FILE}")
print(f"Summary: {summary_file}")
print()
print("IMPORTANT:")
print("The SNP order recorded here must match")
print("the genotype and LD matrix order later.")