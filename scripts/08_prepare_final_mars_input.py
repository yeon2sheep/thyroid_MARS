from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

GENO_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "matched"
    / "chr22_EAS504_genotype_qc.tsv"
)

Z_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "matched"
    / "chr22_gwas_z_qc.tsv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "mars_input"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FINAL_GENO_FILE = (
    OUTPUT_DIR
    / "chr22_EAS504_genotype_final.tsv"
)

FINAL_Z_FILE = (
    OUTPUT_DIR
    / "chr22_gwas_z_final.tsv"
)

FINAL_SNP_FILE = (
    OUTPUT_DIR
    / "chr22_final_snps.tsv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "chr22_mars_input_summary.tsv"
)


# ============================================================
# 2. Load data
# ============================================================

print("=" * 60)
print("Loading QC-passed MARS input")
print("=" * 60)

geno = pd.read_csv(
    GENO_FILE,
    sep="\t"
)

z_df = pd.read_csv(
    Z_FILE,
    sep="\t"
)

print(f"Genotype SNPs : {len(geno):,}")
print(f"Z-score SNPs  : {len(z_df):,}")


# ============================================================
# 3. Basic checks
# ============================================================

assert "variant_key" in geno.columns
assert "variant_key" in z_df.columns

assert geno["variant_key"].is_unique
assert z_df["variant_key"].is_unique

assert (
    geno["variant_key"].tolist()
    == z_df["variant_key"].tolist()
)

print("✓ Genotype and Z-score SNP order initially match")


# ============================================================
# 4. Extract genotype matrix
# ============================================================

sample_columns = geno.columns[1:]

geno_numeric = geno[sample_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

n_samples = len(sample_columns)

print(f"Samples       : {n_samples:,}")

assert n_samples == 504


# ============================================================
# 5. Check missing genotype
# ============================================================

missing_count = geno_numeric.isna().sum().sum()

print(f"Missing genotype : {missing_count:,}")

assert missing_count == 0

print("✓ No missing genotype")


# ============================================================
# 6. Identify monomorphic SNPs
# ============================================================

genotype_std = geno_numeric.std(axis=1)

monomorphic_mask = genotype_std == 0

monomorphic_count = monomorphic_mask.sum()

print("\n" + "=" * 60)
print("Monomorphic SNP QC")
print("=" * 60)

print(
    f"Total SNPs          : {len(geno):,}"
)

print(
    f"Monomorphic SNPs    : {monomorphic_count:,}"
)

print(
    f"Polymorphic SNPs    : "
    f"{(~monomorphic_mask).sum():,}"
)


# ============================================================
# 7. Save removed monomorphic SNP list
# ============================================================

removed_monomorphic = geno.loc[
    monomorphic_mask,
    ["variant_key"]
].copy()

removed_monomorphic["reason"] = "monomorphic"

removed_monomorphic_file = (
    OUTPUT_DIR
    / "chr22_removed_monomorphic_snps.tsv"
)

removed_monomorphic.to_csv(
    removed_monomorphic_file,
    sep="\t",
    index=False
)


# ============================================================
# 8. Keep polymorphic SNPs
# ============================================================

keep_mask = ~monomorphic_mask

geno_final = geno.loc[
    keep_mask
].copy()

z_final = z_df.loc[
    keep_mask
].copy()


# ============================================================
# 9. Final order validation
# ============================================================

assert (
    geno_final["variant_key"].tolist()
    == z_final["variant_key"].tolist()
)

print("\n✓ Monomorphic SNPs removed from genotype")
print("✓ Same SNPs removed from Z-score")
print("✓ SNP order remains identical")


# ============================================================
# 10. Final genotype validation
# ============================================================

final_geno_numeric = geno_final[
    sample_columns
].apply(
    pd.to_numeric,
    errors="coerce"
)

assert final_geno_numeric.isna().sum().sum() == 0

assert final_geno_numeric.min().min() >= 0
assert final_geno_numeric.max().max() <= 2

final_std = final_geno_numeric.std(axis=1)

assert (final_std > 0).all()

print("✓ Final genotype has no missing values")
print("✓ Final genotype values are within 0~2")
print("✓ All final SNPs are polymorphic")


# ============================================================
# 11. Validate Z-score
# ============================================================

z_numeric = pd.to_numeric(
    z_final["z"],
    errors="coerce"
)

assert z_numeric.isna().sum() == 0
assert np.isinf(z_numeric).sum() == 0

print("✓ Final Z-scores contain no NaN or Inf")


# ============================================================
# 12. Save final genotype
# ============================================================

geno_final.to_csv(
    FINAL_GENO_FILE,
    sep="\t",
    index=False
)


# ============================================================
# 13. Save final Z-score
# ============================================================

z_final.to_csv(
    FINAL_Z_FILE,
    sep="\t",
    index=False
)


# ============================================================
# 14. Create final SNP information table
# ============================================================

snp_columns = [
    col
    for col in [
        "variant_key",
        "chromosome",
        "position",
        "effect_allele",
        "other_allele",
        "vcf_ref",
        "vcf_alt",
        "orientation",
    ]
    if col in z_final.columns or col in geno_final.columns
]

# variant_key는 반드시 포함
if "variant_key" not in snp_columns:
    snp_columns.insert(0, "variant_key")

# matched information is not directly available here,
# so create a simple SNP index table.

final_snps = pd.DataFrame({
    "snp_index": np.arange(
        1,
        len(geno_final) + 1
    ),
    "variant_key": geno_final["variant_key"].values
})

final_snps.to_csv(
    FINAL_SNP_FILE,
    sep="\t",
    index=False
)


# ============================================================
# 15. Summary
# ============================================================

summary = pd.DataFrame({
    "metric": [
        "input_qc_snps",
        "removed_monomorphic_snps",
        "final_snps",
        "samples",
        "missing_genotypes",
    ],
    "value": [
        len(geno),
        int(monomorphic_count),
        len(geno_final),
        n_samples,
        int(final_geno_numeric.isna().sum().sum()),
    ]
})

summary.to_csv(
    SUMMARY_FILE,
    sep="\t",
    index=False
)


# ============================================================
# 16. Final result
# ============================================================

print("\n" + "=" * 60)
print("FINAL MARS INPUT PREPARATION")
print("=" * 60)

print(
    f"Input SNPs       : {len(geno):,}"
)

print(
    f"Removed          : {monomorphic_count:,}"
)

print(
    f"Final SNPs       : {len(geno_final):,}"
)

print(
    f"EAS samples      : {n_samples:,}"
)

print(
    f"Final matrix     : "
    f"{len(geno_final):,} × {n_samples:,}"
)

print("\nOutput files:")
print(FINAL_GENO_FILE)
print(FINAL_Z_FILE)
print(FINAL_SNP_FILE)
print(SUMMARY_FILE)
print(removed_monomorphic_file)

print("\n✓ Final MARS input preparation completed")