import subprocess
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

GWAS_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "GCST90627762_chr22_prepared.tsv"
)

REFERENCE_VCF = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "reference"
    / "chr22_EAS504_reference.vcf.gz"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "harmonized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHED_FILE = OUTPUT_DIR / "chr22_matched_snps.tsv"
Z_FILE = OUTPUT_DIR / "chr22_gwas_z.tsv"
GENOTYPE_FILE = OUTPUT_DIR / "chr22_EAS504_genotype.tsv"
SUMMARY_FILE = OUTPUT_DIR / "chr22_harmonization_summary.tsv"


# ============================================================
# 2. Settings
# ============================================================

# Only biallelic SNPs are expected from the reference VCF.
VALID_BASES = {"A", "C", "G", "T"}

COMPLEMENT = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
}


# ============================================================
# 3. Helper functions
# ============================================================

def complement_allele(allele):
    """Return DNA strand complement."""
    allele = str(allele).upper()

    if allele not in COMPLEMENT:
        return None

    return COMPLEMENT[allele]


def is_palindromic(ref, alt):
    """
    Check whether a SNP is palindromic.

    Palindromic SNPs:
        A/T
        T/A
        C/G
        G/C
    """
    pair = {str(ref).upper(), str(alt).upper()}

    return pair in [
        {"A", "T"},
        {"C", "G"},
    ]


def classify_alleles(effect, other, ref, alt):
    """
    Determine how GWAS alleles correspond to VCF REF/ALT.

    Returns:
        orientation
        dosage_mode

    orientation:
        ALT_IS_EFFECT
        REF_IS_EFFECT
        ALT_IS_EFFECT_COMPLEMENT
        REF_IS_EFFECT_COMPLEMENT
        PALINDROMIC_COMPLEMENT_UNRESOLVED
        ALLELE_MISMATCH

    dosage_mode:
        ALT
        REF
        None
    """

    effect = str(effect).upper()
    other = str(other).upper()
    ref = str(ref).upper()
    alt = str(alt).upper()

    # --------------------------------------------------------
    # 1. Exact direct match
    # --------------------------------------------------------
    if effect == alt and other == ref:
        return "ALT_IS_EFFECT", "ALT"

    # --------------------------------------------------------
    # 2. Exact reverse match
    # --------------------------------------------------------
    if effect == ref and other == alt:
        return "REF_IS_EFFECT", "REF"

    # --------------------------------------------------------
    # 3. Strand complement
    # --------------------------------------------------------

    # Do not use complement-only matching for palindromic SNPs.
    if is_palindromic(ref, alt):
        return "PALINDROMIC_COMPLEMENT_UNRESOLVED", None

    effect_comp = complement_allele(effect)
    other_comp = complement_allele(other)

    if effect_comp is None or other_comp is None:
        return "ALLELE_MISMATCH", None

    # GWAS EA/OA after strand flip corresponds to VCF ALT/REF
    if effect_comp == alt and other_comp == ref:
        return "ALT_IS_EFFECT_COMPLEMENT", "ALT"

    # GWAS EA/OA after strand flip corresponds to VCF REF/ALT
    if effect_comp == ref and other_comp == alt:
        return "REF_IS_EFFECT_COMPLEMENT", "REF"

    return "ALLELE_MISMATCH", None


def parse_gt(gt):
    """
    Convert a VCF genotype string into ALT dosage.

    Examples:
        0/0 -> 0
        0/1 -> 1
        1/0 -> 1
        1/1 -> 2
        0|0 -> 0
        0|1 -> 1
        1|0 -> 1
        1|1 -> 2

    Missing:
        ./.
        .|.
        .
        -> np.nan
    """

    if gt is None:
        return np.nan

    gt = str(gt)

    if gt in {".", "./.", ".|."}:
        return np.nan

    # Handle possible FORMAT-like strings just in case.
    if ":" in gt:
        gt = gt.split(":")[0]

    gt = gt.replace("|", "/")

    if gt in {".", "./."}:
        return np.nan

    alleles = gt.split("/")

    if len(alleles) != 2:
        return np.nan

    if "." in alleles:
        return np.nan

    try:
        a1 = int(alleles[0])
        a2 = int(alleles[1])
    except ValueError:
        return np.nan

    # Reference VCF has already been restricted to biallelic SNPs.
    if a1 not in {0, 1} or a2 not in {0, 1}:
        return np.nan

    return a1 + a2


# ============================================================
# 4. Read GWAS
# ============================================================

print("=" * 70)
print("03. GWAS ↔ REFERENCE HARMONIZATION")
print("=" * 70)

print(f"GWAS input     : {GWAS_FILE}")
print(f"Reference VCF  : {REFERENCE_VCF}")
print()

gwas = pd.read_csv(GWAS_FILE, sep="\t")

print(f"GWAS SNPs: {len(gwas):,}")


# ------------------------------------------------------------
# Basic cleanup
# ------------------------------------------------------------

gwas["chromosome"] = (
    gwas["chromosome"]
    .astype(str)
    .str.replace("^chr", "", regex=True)
)

gwas["base_pair_location"] = pd.to_numeric(
    gwas["base_pair_location"],
    errors="coerce",
)

gwas["effect_allele"] = (
    gwas["effect_allele"]
    .astype(str)
    .str.upper()
)

gwas["other_allele"] = (
    gwas["other_allele"]
    .astype(str)
    .str.upper()
)

gwas["z"] = pd.to_numeric(
    gwas["z"],
    errors="coerce",
)

gwas = gwas.dropna(
    subset=[
        "chromosome",
        "base_pair_location",
        "effect_allele",
        "other_allele",
        "z",
    ]
).copy()

gwas["base_pair_location"] = (
    gwas["base_pair_location"]
    .astype(int)
)


# ============================================================
# 5. Read reference VCF metadata
# ============================================================

print()
print("Reading reference VCF variants...")

query_format = "%CHROM\t%POS\t%REF\t%ALT\n"

result = subprocess.run(
    [
        "bcftools",
        "query",
        "-f",
        query_format,
        str(REFERENCE_VCF),
    ],
    capture_output=True,
    text=True,
    check=True,
)

vcf_rows = []

for line in result.stdout.splitlines():

    if not line.strip():
        continue

    fields = line.split("\t")

    if len(fields) != 4:
        continue

    chrom, pos, ref, alt = fields

    chrom = chrom.replace("chr", "")
    pos = int(pos)
    ref = ref.upper()
    alt = alt.upper()

    # Keep only simple A/C/G/T SNPs.
    if (
        len(ref) != 1
        or len(alt) != 1
        or ref not in VALID_BASES
        or alt not in VALID_BASES
    ):
        continue

    vcf_rows.append(
        {
            "chromosome": chrom,
            "base_pair_location": pos,
            "vcf_ref": ref,
            "vcf_alt": alt,
        }
    )


vcf = pd.DataFrame(vcf_rows)

print(f"Reference SNPs: {len(vcf):,}")


# ============================================================
# 6. Match by chromosome + position
# ============================================================

print()
print("Matching SNP positions...")

# Reference VCF should not contain duplicate records after
# biallelic SNP filtering, but keep only one if duplicates occur.
vcf = vcf.drop_duplicates(
    subset=["chromosome", "base_pair_location"],
    keep="first",
).copy()

merged = gwas.merge(
    vcf,
    on=["chromosome", "base_pair_location"],
    how="left",
    indicator=True,
)

position_matched = merged["_merge"].eq("both")

print(
    f"Position matched: "
    f"{position_matched.sum():,} / {len(merged):,}"
)

print(
    f"Position not found: "
    f"{(~position_matched).sum():,}"
)

merged = merged.drop(columns=["_merge"])


# ============================================================
# 7. Allele harmonization
# ============================================================

print()
print("Checking allele orientation...")

orientations = []
dosage_modes = []

for _, row in merged.iterrows():

    if pd.isna(row["vcf_ref"]) or pd.isna(row["vcf_alt"]):

        orientations.append("POSITION_NOT_FOUND")
        dosage_modes.append(None)

        continue

    orientation, dosage_mode = classify_alleles(
        row["effect_allele"],
        row["other_allele"],
        row["vcf_ref"],
        row["vcf_alt"],
    )

    orientations.append(orientation)
    dosage_modes.append(dosage_mode)


merged["orientation"] = orientations
merged["dosage_mode"] = dosage_modes


# ============================================================
# 8. Print harmonization summary
# ============================================================

orientation_counts = (
    merged["orientation"]
    .value_counts(dropna=False)
)

print()
print("-" * 70)
print("HARMONIZATION SUMMARY")
print("-" * 70)

for name, count in orientation_counts.items():

    print(
        f"{str(name):40s} "
        f"{int(count):,}"
    )


# ============================================================
# 9. Keep only successfully harmonized SNPs
# ============================================================

valid_orientations = {
    "ALT_IS_EFFECT",
    "REF_IS_EFFECT",
    "ALT_IS_EFFECT_COMPLEMENT",
    "REF_IS_EFFECT_COMPLEMENT",
}

matched = merged[
    merged["orientation"].isin(valid_orientations)
].copy()

print()
print(f"Successfully harmonized SNPs: {len(matched):,}")


if len(matched) == 0:
    raise RuntimeError(
        "No SNPs could be harmonized. "
        "Check chromosome/position/allele representation."
    )


# ============================================================
# 10. Create stable variant key
# ============================================================

matched["variant_key"] = (
    matched["chromosome"].astype(str)
    + ":"
    + matched["base_pair_location"].astype(str)
    + ":"
    + matched["effect_allele"]
    + ":"
    + matched["other_allele"]
)


# ============================================================
# 11. Extract genotype from reference VCF
# ============================================================

print()
print("Extracting EAS genotype data...")
print("This may take some time.")

query_format = (
    "%CHROM\t%POS\t%REF\t%ALT"
    "[\t%GT]\n"
)

result = subprocess.run(
    [
        "bcftools",
        "query",
        "-f",
        query_format,
        str(REFERENCE_VCF),
    ],
    capture_output=True,
    text=True,
    check=True,
)

reference_samples_result = subprocess.run(
    [
        "bcftools",
        "query",
        "-l",
        str(REFERENCE_VCF),
    ],
    capture_output=True,
    text=True,
    check=True,
)

samples = [
    x.strip()
    for x in reference_samples_result.stdout.splitlines()
    if x.strip()
]

print(f"Reference samples: {len(samples):,}")


if len(samples) != 504:
    print(
        f"WARNING: Expected 504 EAS samples, "
        f"but found {len(samples):,}."
    )


# ============================================================
# 12. Build reference genotype lookup
# ============================================================

print("Building genotype lookup...")

genotype_lookup = {}

for line in result.stdout.splitlines():

    if not line.strip():
        continue

    fields = line.split("\t")

    if len(fields) < 4:
        continue

    chrom = fields[0].replace("chr", "")
    pos = int(fields[1])
    ref = fields[2].upper()
    alt = fields[3].upper()

    key = (
        chrom,
        pos,
        ref,
        alt,
    )

    gt_values = fields[4:]

    if len(gt_values) != len(samples):
        raise RuntimeError(
            f"Sample count mismatch at "
            f"{chrom}:{pos}:{ref}:{alt}. "
            f"Expected {len(samples)}, "
            f"got {len(gt_values)}."
        )

    genotype_lookup[key] = gt_values


# ============================================================
# 13. Construct genotype matrix in GWAS order
# ============================================================

print("Constructing effect-allele dosage matrix...")

genotype_rows = []
successful_rows = []

missing_reference_genotypes = 0

for _, row in matched.iterrows():

    key = (
        str(row["chromosome"]),
        int(row["base_pair_location"]),
        str(row["vcf_ref"]),
        str(row["vcf_alt"]),
    )

    gt_values = genotype_lookup.get(key)

    if gt_values is None:

        missing_reference_genotypes += 1
        continue

    alt_dosage = np.array(
        [
            parse_gt(gt)
            for gt in gt_values
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # Convert ALT dosage to GWAS effect-allele dosage.
    # --------------------------------------------------------

    if row["dosage_mode"] == "ALT":

        effect_dosage = alt_dosage

    elif row["dosage_mode"] == "REF":

        effect_dosage = 2.0 - alt_dosage

    else:

        raise RuntimeError(
            f"Unexpected dosage mode: "
            f"{row['dosage_mode']}"
        )

    genotype_rows.append(effect_dosage)
    successful_rows.append(row)


if missing_reference_genotypes > 0:

    print(
        "WARNING: "
        f"{missing_reference_genotypes:,} matched variants "
        "were missing from genotype lookup."
    )


matched = pd.DataFrame(successful_rows)

genotype_matrix = np.vstack(genotype_rows)


# ============================================================
# 14. Final consistency checks
# ============================================================

print()
print("Running consistency checks...")

if len(matched) != genotype_matrix.shape[0]:
    raise RuntimeError(
        "Variant count mismatch between metadata "
        "and genotype matrix."
    )

if genotype_matrix.shape[1] != len(samples):
    raise RuntimeError(
        "Sample count mismatch in genotype matrix."
    )

if not np.all(np.isfinite(
    matched["z"].to_numpy(dtype=float)
)):
    raise RuntimeError(
        "Non-finite Z-scores remain."
    )

if np.nanmin(genotype_matrix) < 0:
    raise RuntimeError(
        "Genotype dosage below 0 detected."
    )

if np.nanmax(genotype_matrix) > 2:
    raise RuntimeError(
        "Genotype dosage above 2 detected."
    )


# ============================================================
# 15. Save matched SNP metadata
# ============================================================

matched_columns = [
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
    matched_columns
].to_csv(
    MATCHED_FILE,
    sep="\t",
    index=False,
)


# ============================================================
# 16. Save Z-score vector
# ============================================================

z_output = matched[
    [
        "variant_key",
        "z",
    ]
].copy()

z_output.to_csv(
    Z_FILE,
    sep="\t",
    index=False,
)


# ============================================================
# 17. Save genotype matrix
# ============================================================

genotype_output = pd.DataFrame(
    genotype_matrix,
    columns=samples,
)

genotype_output.insert(
    0,
    "variant_key",
    matched["variant_key"].tolist(),
)

genotype_output.to_csv(
    GENOTYPE_FILE,
    sep="\t",
    index=False,
)


# ============================================================
# 18. Save summary
# ============================================================

summary_rows = [
    ("GWAS SNPs", len(gwas)),
    ("Reference SNPs", len(vcf)),
    (
        "Position matched",
        int(position_matched.sum()),
    ),
    (
        "Position not found",
        int((~position_matched).sum()),
    ),
    (
        "Successfully harmonized",
        len(matched),
    ),
    (
        "Genotype samples",
        len(samples),
    ),
    (
        "Missing genotype lookup",
        missing_reference_genotypes,
    ),
    (
        "ALT_IS_EFFECT",
        int(
            (merged["orientation"] == "ALT_IS_EFFECT").sum()
        ),
    ),
    (
        "REF_IS_EFFECT",
        int(
            (merged["orientation"] == "REF_IS_EFFECT").sum()
        ),
    ),
    (
        "ALT_IS_EFFECT_COMPLEMENT",
        int(
            (
                merged["orientation"]
                == "ALT_IS_EFFECT_COMPLEMENT"
            ).sum()
        ),
    ),
    (
        "REF_IS_EFFECT_COMPLEMENT",
        int(
            (
                merged["orientation"]
                == "REF_IS_EFFECT_COMPLEMENT"
            ).sum()
        ),
    ),
    (
        "PALINDROMIC_COMPLEMENT_UNRESOLVED",
        int(
            (
                merged["orientation"]
                == "PALINDROMIC_COMPLEMENT_UNRESOLVED"
            ).sum()
        ),
    ),
    (
        "ALLELE_MISMATCH",
        int(
            (
                merged["orientation"]
                == "ALLELE_MISMATCH"
            ).sum()
        ),
    ),
]

summary = pd.DataFrame(
    summary_rows,
    columns=["metric", "value"],
)

summary.to_csv(
    SUMMARY_FILE,
    sep="\t",
    index=False,
)


# ============================================================
# 19. Final report
# ============================================================

print()
print("=" * 70)
print("HARMONIZATION COMPLETE")
print("=" * 70)

print(f"Final matched SNPs : {len(matched):,}")
print(f"EAS samples        : {len(samples):,}")
print()
print(f"Matched SNPs       : {MATCHED_FILE}")
print(f"Z-scores           : {Z_FILE}")
print(f"Genotype matrix    : {GENOTYPE_FILE}")
print(f"Summary            : {SUMMARY_FILE}")

print()
print("Genotype orientation:")
print("  ALT_IS_EFFECT              -> ALT dosage")
print("  REF_IS_EFFECT              -> 2 - ALT dosage")
print("  ALT_IS_EFFECT_COMPLEMENT   -> ALT dosage")
print("  REF_IS_EFFECT_COMPLEMENT   -> 2 - ALT dosage")

print()
print("Next step:")
print("  python scripts/04_qc_and_ld.py")
print("=" * 70)