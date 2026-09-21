import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "harmonized"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MATCHED_FILE = (
    INPUT_DIR
    / "chr22_matched_snps.tsv"
)

Z_FILE = (
    INPUT_DIR
    / "chr22_gwas_z.tsv"
)

GENOTYPE_FILE = (
    INPUT_DIR
    / "chr22_EAS504_genotype.tsv"
)


FINAL_SNPS_FILE = (
    OUTPUT_DIR
    / "chr22_final_snps.tsv"
)

FINAL_Z_FILE = (
    OUTPUT_DIR
    / "chr22_final_z.tsv"
)

FINAL_GENOTYPE_FILE = (
    OUTPUT_DIR
    / "chr22_final_genotype.tsv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "chr22_qc_summary.tsv"
)


# ============================================================
# 2. Settings
# ============================================================

MISSING_RATE_THRESHOLD = 0.10
# 한 SNP에서 genotype이 10% 보다 많이 missing되면 그 SNP 제거하는 그 기준
# EAS 샘플이 504명이니까 약 51명 이상 genotype이 미싱되는 SNP 제거

# ============================================================
# 3. Start
# ============================================================

print("=" * 70)
print("04. GENOTYPE QC")
print("=" * 70)

print(f"Matched SNPs : {MATCHED_FILE}")
print(f"Z-scores     : {Z_FILE}")
print(f"Genotype     : {GENOTYPE_FILE}")
print()


# ============================================================
# 4. Read input
# ============================================================

matched = pd.read_csv(
    MATCHED_FILE,
    sep="\t"
)

z_df = pd.read_csv(
    Z_FILE,
    sep="\t"
)

genotype_df = pd.read_csv(
    GENOTYPE_FILE,
    sep="\t"
)
# 3번 코드 결과 읽기

print(f"Input SNPs      : {len(matched):,}")

print(
    f"Input genotype  : "
    f"{genotype_df.shape[0]:,} SNPs × "
    f"{genotype_df.shape[1] - 1:,} samples"
)


# ============================================================
# 5. Check variant order
# ============================================================

if not matched["variant_key"].equals(
    z_df["variant_key"]
):
    raise RuntimeError(
        "Variant order mismatch between matched SNPs "
        "and Z-scores."
    )


if not matched["variant_key"].equals(
    genotype_df["variant_key"]
):
    raise RuntimeError(
        "Variant order mismatch between matched SNPs "
        "and genotype matrix."
    )


# ============================================================
# 6. Extract genotype matrix
# ============================================================

sample_names = (
    genotype_df.columns[1:]
    .tolist()
)

genotype = (
    genotype_df[sample_names]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .to_numpy(
        dtype=float
    )
)


# ============================================================
# 7. Basic genotype validation
# ============================================================

invalid_low = np.sum(
    genotype < 0
)

invalid_high = np.sum(
    genotype > 2
)

if invalid_low > 0 or invalid_high > 0:

    raise RuntimeError(
        "Invalid genotype values detected. "
        f"<0: {invalid_low:,}, "
        f">2: {invalid_high:,}"
    )


# ============================================================
# 8. Missingness QC
# ============================================================

print()
print("-" * 70)
print("MISSINGNESS QC")
print("-" * 70)

missing_count = np.isnan(
    genotype
).sum(axis=1)

missing_rate = (
    missing_count
    / genotype.shape[1]
)

keep = (
    missing_rate
    <= MISSING_RATE_THRESHOLD
)

removed_missing = int(
    (~keep).sum()
)

print(
    f"Threshold                  : "
    f"{MISSING_RATE_THRESHOLD:.0%}"
)

print(
    f"Removed high-missing SNPs  : "
    f"{removed_missing:,}"
)

print(
    f"Remaining SNPs             : "
    f"{keep.sum():,}"
)


# Apply filter

matched = matched.loc[
    keep
].reset_index(drop=True)

genotype = genotype[
    keep
]


# ============================================================
# 9. Mean imputation
# ============================================================

print()
print("Mean imputation...")

imputed_count = 0

for i in range(
    genotype.shape[0]
):

    row = genotype[i]

    missing_mask = np.isnan(row)

    if not missing_mask.any():
        continue

    mean_dosage = np.nanmean(
        row
    )

    if not np.isfinite(
        mean_dosage
    ):
        raise RuntimeError(
            f"All genotypes are missing "
            f"for row {i}."
        )

    genotype[
        i,
        missing_mask
    ] = mean_dosage

    imputed_count += int(
        missing_mask.sum()
    )


print(
    f"Missing values imputed      : "
    f"{imputed_count:,}"
)


# ============================================================
# 10. Remove monomorphic SNPs
# ============================================================

print()
print("Checking monomorphic SNPs...")

genotype_min = np.min(
    genotype,
    axis=1
)

genotype_max = np.max(
    genotype,
    axis=1
)

polymorphic = (
    genotype_max > genotype_min
)

removed_monomorphic = int(
    (~polymorphic).sum()
)

print(
    f"Monomorphic SNPs removed    : "
    f"{removed_monomorphic:,}"
)

print(
    f"Remaining SNPs              : "
    f"{polymorphic.sum():,}"
)


matched = matched.loc[
    polymorphic
].reset_index(drop=True)

genotype = genotype[
    polymorphic
]


# ============================================================
# 11. Final checks
# ============================================================

z_values = matched[
    "z"
].to_numpy(
    dtype=float
)


if not np.all(
    np.isfinite(genotype)
):
    raise RuntimeError(
        "Non-finite genotype values remain."
    )


if not np.all(
    np.isfinite(z_values)
):
    raise RuntimeError(
        "Non-finite Z-scores remain."
    )


if np.any(
    genotype < 0
) or np.any(
    genotype > 2
):
    raise RuntimeError(
        "Genotype values outside 0-2 remain."
    )


if genotype.shape[0] != len(
    matched
):
    raise RuntimeError(
        "Final SNP count mismatch."
    )


# ============================================================
# 12. Save final SNP metadata
# ============================================================

final_columns = [
    "variant_key",
    "chromosome",
    "base_pair_location",
    "effect_allele",
    "other_allele",
    "vcf_ref",
    "vcf_alt",
    "orientation",
    "dosage_mode",
    "beta",
    "standard_error",
    "effect_allele_frequency",
    "p_value",
    "n",
    "z",
]

matched[
    final_columns
].to_csv(
    FINAL_SNPS_FILE,
    sep="\t",
    index=False
)


# ============================================================
# 13. Save final Z-scores
# ============================================================

z_output = pd.DataFrame(
    {
        "variant_key":
            matched["variant_key"].tolist(),

        "z":
            z_values
    }
)

z_output.to_csv(
    FINAL_Z_FILE,
    sep="\t",
    index=False
)


# ============================================================
# 14. Save final genotype
# ============================================================

final_genotype = pd.DataFrame(
    genotype,
    columns=sample_names
)

final_genotype.insert(
    0,
    "variant_key",
    matched["variant_key"].tolist()
)

final_genotype.to_csv(
    FINAL_GENOTYPE_FILE,
    sep="\t",
    index=False
)


# ============================================================
# 15. Save QC summary
# ============================================================

summary_rows = [
    (
        "input_snps",
        len(genotype_df)
    ),

    (
        "input_samples",
        len(sample_names)
    ),

    (
        "removed_high_missingness",
        removed_missing
    ),

    (
        "imputed_genotype_values",
        imputed_count
    ),

    (
        "removed_monomorphic",
        removed_monomorphic
    ),

    (
        "final_snps",
        len(matched)
    ),

    (
        "final_samples",
        len(sample_names)
    ),

    (
        "missingness_threshold",
        MISSING_RATE_THRESHOLD
    ),
]

summary = pd.DataFrame(
    summary_rows,
    columns=[
        "metric",
        "value"
    ]
)

summary.to_csv(
    SUMMARY_FILE,
    sep="\t",
    index=False
)


# ============================================================
# 16. Final report
# ============================================================

print()
print("=" * 70)
print("GENOTYPE QC COMPLETE")
print("=" * 70)

print(
    f"Final SNPs      : "
    f"{len(matched):,}"
)

print(
    f"EAS samples     : "
    f"{len(sample_names):,}"
)

print(
    f"Final genotype  : "
    f"{genotype.shape[0]:,} × "
    f"{genotype.shape[1]:,}"
)

print()
print(f"Final SNPs      : {FINAL_SNPS_FILE}")
print(f"Final Z-scores  : {FINAL_Z_FILE}")
print(f"Final genotype  : {FINAL_GENOTYPE_FILE}")
print(f"QC summary      : {SUMMARY_FILE}")

print()
print("LD calculation will be performed separately")
print("for each analysis bin in script 05.")

print()
print("Next step:")
print("  python scripts/05_prepare_mars_inputs.py")

print("=" * 70)