import subprocess
from pathlib import Path

import pandas as pd


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

MARS_SNP_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "mars_input"
    / "chr22_test_mars_snps.tsv"
)

VCF_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "reference"
    / "chr22_EAS504_testSNPs.vcf.gz"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "matched"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MATCHED_FILE = OUTPUT_DIR / "chr22_matched_snps.tsv"
Z_FILE = OUTPUT_DIR / "chr22_gwas_z.tsv"
GENOTYPE_FILE = OUTPUT_DIR / "chr22_EAS504_genotype.tsv"
SUMMARY_FILE = OUTPUT_DIR / "chr22_match_summary.tsv"


# ============================================================
# 2. Helper functions
# ============================================================

def normalize_allele(allele):
    """Uppercase and strip whitespace."""
    return str(allele).strip().upper()


def dosage_from_gt(gt):
    """
    Convert a diploid GT to ALT allele dosage.

    0/0 -> 0
    0/1 -> 1
    1/0 -> 1
    1/1 -> 2

    Missing / unexpected genotypes -> None
    """

    if gt in {"0/0", "0|0"}:
        return 0

    if gt in {"0/1", "1/0", "0|1", "1|0"}:
        return 1

    if gt in {"1/1", "1|1"}:
        return 2

    return None


# ============================================================
# 3. Load GWAS / MARS SNP table
# ============================================================

print("=" * 70)
print("STEP 1: Loading GWAS / MARS SNP table")
print("=" * 70)

mars = pd.read_csv(MARS_SNP_FILE, sep="\t")

print(f"GWAS/MARS records: {len(mars):,}")


# Normalize column names
mars["chromosome"] = (
    mars["chromosome"]
    .astype(str)
    .str.replace("chr", "", regex=False)
)

mars["base_pair_location"] = mars["base_pair_location"].astype(int)

mars["effect_allele_x"] = (
    mars["effect_allele_x"]
    .map(normalize_allele)
)

mars["other_allele_x"] = (
    mars["other_allele_x"]
    .map(normalize_allele)
)


# ============================================================
# 4. Load VCF variant information
# ============================================================

print()
print("=" * 70)
print("STEP 2: Reading 1000 Genomes VCF")
print("=" * 70)

variant_cmd = [
    "bcftools",
    "query",
    "-f",
    "%CHROM\\t%POS\\t%REF\\t%ALT\\n",
    str(VCF_FILE),
]

result = subprocess.run(
    variant_cmd,
    capture_output=True,
    text=True,
    check=True,
)

vcf_rows = []

for line in result.stdout.splitlines():

    if not line.strip():
        continue

    chrom, pos, ref, alt = line.split("\t")

    # This script expects biallelic SNPs.
    if len(ref) != 1 or len(alt) != 1:
        continue

    vcf_rows.append(
        {
            "vcf_chromosome": chrom.replace("chr", ""),
            "vcf_position": int(pos),
            "vcf_ref": normalize_allele(ref),
            "vcf_alt": normalize_allele(alt),
        }
    )


vcf = pd.DataFrame(vcf_rows)

print(f"VCF SNP records: {len(vcf):,}")


# ============================================================
# 5. Match GWAS SNPs to VCF by chromosome + position
# ============================================================

print()
print("=" * 70)
print("STEP 3: Matching GWAS SNPs to VCF")
print("=" * 70)

merged = mars.merge(
    vcf,
    left_on=["chromosome", "base_pair_location"],
    right_on=["vcf_chromosome", "vcf_position"],
    how="left",
)


# ============================================================
# 6. Determine allele orientation
# ============================================================

def determine_orientation(row):

    ea = row["effect_allele_x"]
    oa = row["other_allele_x"]
    ref = row["vcf_ref"]
    alt = row["vcf_alt"]

    if pd.isna(ref) or pd.isna(alt):
        return "NO_VCF"

    # --------------------------------------------------------
    # Case 1:
    # GWAS effect allele = VCF ALT
    #
    # Example:
    # GWAS C/T
    # VCF  T/C
    #
    # ALT dosage already represents effect allele dosage.
    # --------------------------------------------------------

    if ea == alt and oa == ref:
        return "ALT_IS_EFFECT"

    # --------------------------------------------------------
    # Case 2:
    # GWAS effect allele = VCF REF
    #
    # Example:
    # GWAS T/C
    # VCF  T/C
    #
    # VCF dosage counts ALT (C),
    # so effect-allele dosage = 2 - ALT dosage.
    # --------------------------------------------------------

    if ea == ref and oa == alt:
        return "REF_IS_EFFECT"

    # --------------------------------------------------------
    # Otherwise allele mismatch
    # --------------------------------------------------------

    return "ALLELE_MISMATCH"


merged["orientation"] = merged.apply(
    determine_orientation,
    axis=1,
)


# ============================================================
# 7. Keep only successfully matched SNPs
# ============================================================

matched = merged[
    merged["orientation"].isin(
        [
            "ALT_IS_EFFECT",
            "REF_IS_EFFECT",
        ]
    )
].copy()


print(f"Matched SNP records: {len(matched):,}")

print()
print("Orientation counts:")
print(matched["orientation"].value_counts())


# ============================================================
# 8. Extract genotype matrix from VCF
# ============================================================

print()
print("=" * 70)
print("STEP 4: Extracting EAS genotype matrix")
print("=" * 70)

# Extract:
# CHROM
# POS
# REF
# ALT
# followed by GT for all 504 samples

genotype_cmd = [
    "bcftools",
    "query",
    "-f",
    "%CHROM\\t%POS\\t%REF\\t%ALT[\\t%GT]\\n",
    str(VCF_FILE),
]

result = subprocess.run(
    genotype_cmd,
    capture_output=True,
    text=True,
    check=True,
)

genotype_rows = []

for line in result.stdout.splitlines():

    if not line.strip():
        continue

    fields = line.split("\t")

    chrom = fields[0].replace("chr", "")
    pos = int(fields[1])
    ref = normalize_allele(fields[2])
    alt = normalize_allele(fields[3])

    genotypes = fields[4:]

    genotype_rows.append(
        (
            chrom,
            pos,
            ref,
            alt,
            genotypes,
        )
    )


print(f"VCF genotype records: {len(genotype_rows):,}")


# ============================================================
# 9. Build genotype lookup
# ============================================================

genotype_lookup = {}

for chrom, pos, ref, alt, genotypes in genotype_rows:

    key = (chrom, pos, ref, alt)

    genotype_lookup[key] = genotypes


# ============================================================
# 10. Get sample IDs
# ============================================================

sample_cmd = [
    "bcftools",
    "query",
    "-l",
    str(VCF_FILE),
]

result = subprocess.run(
    sample_cmd,
    capture_output=True,
    text=True,
    check=True,
)

sample_ids = result.stdout.strip().splitlines()

print(f"EAS samples: {len(sample_ids):,}")


# ============================================================
# 11. Convert GT -> effect-allele dosage
# ============================================================

print()
print("=" * 70)
print("STEP 5: Converting genotype to effect-allele dosage")
print("=" * 70)

final_rows = []
z_rows = []
genotype_rows_output = []

for _, row in matched.iterrows():

    chrom = row["chromosome"]
    pos = int(row["base_pair_location"])
    ref = row["vcf_ref"]
    alt = row["vcf_alt"]

    key = (chrom, pos, ref, alt)

    if key not in genotype_lookup:
        continue

    raw_genotypes = genotype_lookup[key]

    dosage = []

    for gt in raw_genotypes:

        alt_dosage = dosage_from_gt(gt)

        if alt_dosage is None:
            dosage.append("NA")
            continue

        if row["orientation"] == "ALT_IS_EFFECT":

            # ALT = GWAS effect allele
            effect_dosage = alt_dosage

        elif row["orientation"] == "REF_IS_EFFECT":

            # REF = GWAS effect allele
            effect_dosage = 2 - alt_dosage

        else:
            effect_dosage = None

        dosage.append(effect_dosage)

    # --------------------------------------------------------
    # Keep the SNP information
    # --------------------------------------------------------

    final_rows.append(
        {
            "snp_index": len(final_rows) + 1,
            "bin_id": row["bin_id"],
            "chromosome": chrom,
            "position": pos,
            "effect_allele": row["effect_allele_x"],
            "other_allele": row["other_allele_x"],
            "vcf_ref": ref,
            "vcf_alt": alt,
            "orientation": row["orientation"],
            "variant_key": row["variant_key"],
            "z": row["z"],
            "p_value": row["p_value"],
            "beta": row["beta"],
            "standard_error": row["standard_error"],
        }
    )

    # Z-score output
    z_rows.append(
        {
            "snp_index": len(z_rows) + 1,
            "variant_key": row["variant_key"],
            "z": row["z"],
        }
    )

    # Genotype output
    genotype_rows_output.append(
        [row["variant_key"]] + dosage
    )


# ============================================================
# 12. Create output DataFrames
# ============================================================

matched_final = pd.DataFrame(final_rows)

z_final = pd.DataFrame(z_rows)

genotype_final = pd.DataFrame(
    genotype_rows_output,
    columns=["variant_key"] + sample_ids,
)


# ============================================================
# 13. Save outputs
# ============================================================

print()
print("=" * 70)
print("STEP 6: Saving matched data")
print("=" * 70)

matched_final.to_csv(
    MATCHED_FILE,
    sep="\t",
    index=False,
)

z_final.to_csv(
    Z_FILE,
    sep="\t",
    index=False,
)

genotype_final.to_csv(
    GENOTYPE_FILE,
    sep="\t",
    index=False,
    na_rep="NA",
)


# ============================================================
# 14. Save matching summary
# ============================================================

total_records = len(merged)

no_vcf = (merged["orientation"] == "NO_VCF").sum()

allele_mismatch = (
    merged["orientation"] == "ALLELE_MISMATCH"
).sum()

alt_is_effect = (
    merged["orientation"] == "ALT_IS_EFFECT"
).sum()

ref_is_effect = (
    merged["orientation"] == "REF_IS_EFFECT"
).sum()

summary = pd.DataFrame(
    [
        {
            "gwas_mars_records": len(mars),
            "vcf_snp_records": len(vcf),
            "position_matched_records": (
                total_records - no_vcf
            ),
            "no_vcf_match": no_vcf,
            "allele_mismatch": allele_mismatch,
            "alt_is_effect": alt_is_effect,
            "ref_is_effect": ref_is_effect,
            "final_matched_records": len(matched_final),
            "eas_samples": len(sample_ids),
        }
    ]
)

summary.to_csv(
    SUMMARY_FILE,
    sep="\t",
    index=False,
)


# ============================================================
# 15. Final checks
# ============================================================

print()
print("=" * 70)
print("FINAL CHECK")
print("=" * 70)

print(f"Final matched SNPs : {len(matched_final):,}")
print(f"EAS samples        : {len(sample_ids):,}")
print(f"Z-score rows       : {len(z_final):,}")
print(f"Genotype rows      : {len(genotype_final):,}")
print(f"Genotype columns   : {len(genotype_final.columns):,}")

print()
print("Output files:")
print(MATCHED_FILE)
print(Z_FILE)
print(GENOTYPE_FILE)
print(SUMMARY_FILE)

print()
print("Done.")