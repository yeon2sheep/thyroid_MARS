import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

MATCHED_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "matched"
    / "chr22_matched_snps.tsv"
)

Z_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "matched"
    / "chr22_gwas_z.tsv"
)

GENO_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "matched"
    / "chr22_EAS504_genotype.tsv"
)

OUTPUT_GENO_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "matched"
    / "chr22_EAS504_genotype_qc.tsv"
)

OUTPUT_Z_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "matched"
    / "chr22_gwas_z_qc.tsv"
)

QC_SUMMARY_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "matched"
    / "chr22_genotype_qc_summary.tsv"
)


# ============================================================
# 2. Parameters
# ============================================================

MISSING_RATE_THRESHOLD = 0.10


# ============================================================
# 3. Load data
# ============================================================

print("=" * 60)
print("Loading MARS input files")
print("=" * 60)

matched = pd.read_csv(MATCHED_FILE, sep="\t")
z_df = pd.read_csv(Z_FILE, sep="\t")
geno = pd.read_csv(GENO_FILE, sep="\t")

print(f"Matched SNP records : {len(matched):,}")
print(f"Z-score records     : {len(z_df):,}")
print(f"Genotype records    : {len(geno):,}")
print(f"Genotype columns    : {geno.shape[1]}")


# ============================================================
# 4. Basic checks
# ============================================================

assert "variant_key" in matched.columns
assert "variant_key" in z_df.columns
assert "variant_key" in geno.columns

assert matched["variant_key"].is_unique
assert z_df["variant_key"].is_unique
assert geno["variant_key"].is_unique

assert len(matched) == len(z_df) == len(geno)

assert (
    matched["variant_key"].tolist()
    == z_df["variant_key"].tolist()
    == geno["variant_key"].tolist()
)

print("\n✓ variant_key exists in all files")
print("✓ SNP counts are identical")
print("✓ No duplicate variant_key")
print("✓ SNP order is identical across all files")


# ============================================================
# 5. Validate Z-scores
# ============================================================

z = pd.to_numeric(z_df["z"], errors="coerce")

nan_z = z.isna().sum()
inf_z = np.isinf(z).sum()

print(f"\nNaN Z-scores : {nan_z}")
print(f"Inf Z-scores : {inf_z}")

assert nan_z == 0
assert inf_z == 0

print("✓ Z-scores contain no NaN or Inf")


# ============================================================
# 6. Convert genotype columns to numeric
# ============================================================

sample_columns = geno.columns[1:]

geno_numeric = geno[sample_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

n_snps = len(geno_numeric)
n_samples = len(sample_columns)

print(f"\nSNPs    : {n_snps:,}")
print(f"Samples : {n_samples:,}")

assert n_samples == 504

print("✓ Genotype matrix = SNPs × 504 EAS samples")


# ============================================================
# 7. Calculate SNP missingness
# ============================================================

missing_count = geno_numeric.isna().sum(axis=1)
missing_rate = missing_count / n_samples

total_missing = missing_count.sum()
total_genotypes = n_snps * n_samples
overall_missing_rate = total_missing / total_genotypes

print("\n" + "=" * 60)
print("Genotype missingness")
print("=" * 60)

print(f"Total genotype cells : {total_genotypes:,}")
print(f"Missing genotype     : {total_missing:,}")
print(f"Overall missing rate : {overall_missing_rate:.6%}")

print(
    f"\nSNPs with missing genotype : "
    f"{(missing_count > 0).sum():,}"
)

print(
    f"SNPs with >10% missing     : "
    f"{(missing_rate > MISSING_RATE_THRESHOLD).sum():,}"
)


# ============================================================
# 8. SNP-level QC
# ============================================================

keep_mask = missing_rate <= MISSING_RATE_THRESHOLD
remove_mask = ~keep_mask

removed = geno.loc[remove_mask, ["variant_key"]].copy()
removed["missing_count"] = missing_count[remove_mask].values
removed["missing_rate"] = missing_rate[remove_mask].values

removed = removed.sort_values(
    "missing_rate",
    ascending=False
)

print("\n" + "=" * 60)
print("SNP QC")
print("=" * 60)

print(
    f"Original SNPs : {n_snps:,}"
)

print(
    f"Removed SNPs  : {remove_mask.sum():,}"
)

print(
    f"Remaining SNPs: {keep_mask.sum():,}"
)

if len(removed) > 0:
    print("\nRemoved SNPs:")
    print(removed.to_string(index=False))


# ============================================================
# 9. Keep SNPs passing QC
# ============================================================

geno_qc = geno.loc[keep_mask].copy()
geno_qc_numeric = geno_numeric.loc[keep_mask].copy()

z_qc = z_df.loc[keep_mask].copy()
matched_qc = matched.loc[keep_mask].copy()


# ============================================================
# 10. Mean dosage imputation
# ============================================================

print("\n" + "=" * 60)
print("Mean dosage imputation")
print("=" * 60)

imputed_total = 0

for idx in geno_qc_numeric.index:

    row = geno_qc_numeric.loc[idx]

    if row.isna().any():

        mean_dosage = row.mean()

        if pd.isna(mean_dosage):
            raise ValueError(
                f"All genotype values are missing for "
                f"{geno_qc.loc[idx, 'variant_key']}"
            )

        missing_n = row.isna().sum()

        geno_qc_numeric.loc[idx] = row.fillna(mean_dosage)

        imputed_total += missing_n


print(f"Imputed genotype values : {imputed_total:,}")


# ============================================================
# 11. Check genotype values
# ============================================================

remaining_missing = geno_qc_numeric.isna().sum().sum()

print(
    f"Remaining missing values: {remaining_missing:,}"
)

assert remaining_missing == 0

print("✓ No missing genotype remains")


# ============================================================
# 12. Check genotype range
# ============================================================

min_genotype = geno_qc_numeric.min().min()
max_genotype = geno_qc_numeric.max().max()

print(
    f"Genotype range          : "
    f"{min_genotype:.6f} ~ {max_genotype:.6f}"
)

assert min_genotype >= 0
assert max_genotype <= 2

print("✓ All genotype dosages are within 0~2")


# ============================================================
# 13. Check monomorphic SNPs
# ============================================================

genotype_std = geno_qc_numeric.std(axis=1)

monomorphic_mask = genotype_std == 0

monomorphic_count = monomorphic_mask.sum()

print(
    f"\nMonomorphic SNPs        : "
    f"{monomorphic_count:,}"
)

if monomorphic_count > 0:
    print(
        "\nWarning: monomorphic SNPs detected."
    )


# ============================================================
# 14. Rebuild final genotype file
# ============================================================

geno_qc_final = geno_qc[["variant_key"]].copy()

for sample in sample_columns:
    geno_qc_final[sample] = geno_qc_numeric[sample].values


# ============================================================
# 15. Save outputs
# ============================================================

geno_qc_final.to_csv(
    OUTPUT_GENO_FILE,
    sep="\t",
    index=False
)

z_qc.to_csv(
    OUTPUT_Z_FILE,
    sep="\t",
    index=False
)

qc_summary = pd.DataFrame({
    "metric": [
        "original_snps",
        "removed_snps_missing_gt_gt10pct",
        "remaining_snps",
        "samples",
        "total_genotype_cells",
        "original_missing_genotypes",
        "overall_original_missing_rate",
        "imputed_genotypes",
        "remaining_missing_genotypes",
        "monomorphic_snps",
    ],
    "value": [
        n_snps,
        int(remove_mask.sum()),
        int(keep_mask.sum()),
        n_samples,
        total_genotypes,
        int(total_missing),
        overall_missing_rate,
        int(imputed_total),
        int(remaining_missing),
        int(monomorphic_count),
    ]
})

qc_summary.to_csv(
    QC_SUMMARY_FILE,
    sep="\t",
    index=False
)


# ============================================================
# 16. Final validation
# ============================================================

assert len(geno_qc_final) == len(z_qc)
assert len(geno_qc_final) == len(matched_qc)

assert (
    geno_qc_final["variant_key"].tolist()
    == z_qc["variant_key"].tolist()
)

assert geno_qc_final.iloc[:, 1:].isna().sum().sum() == 0

print("\n" + "=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

print(f"Final SNPs    : {len(geno_qc_final):,}")
print(f"Samples       : {n_samples:,}")
print(f"Genotype size : {len(geno_qc_final):,} × {n_samples:,}")

print("\n✓ SNP order preserved")
print("✓ Z-score order preserved")
print("✓ No missing genotype")
print("✓ Genotype values within 0~2")

print("\nOutput files:")
print(OUTPUT_GENO_FILE)
print(OUTPUT_Z_FILE)
print(QC_SUMMARY_FILE)

print("\n✓ QC pipeline completed successfully")