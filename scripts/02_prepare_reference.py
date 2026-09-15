import subprocess
from pathlib import Path

import pandas as pd


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

VCF_FILE = (
    PROJECT_DIR
    / "data"
    / "reference"
    / "chr22.vcf"
)

EAS_SAMPLE_FILE = (
    PROJECT_DIR
    / "data"
    / "reference"
    / "EAS.samples.txt"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "reference"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET_FILE = (
    OUTPUT_DIR
    / "chr22_target_positions.tsv"
)

OUTPUT_VCF = (
    OUTPUT_DIR
    / "chr22_EAS504_reference.vcf.gz"
)


# ============================================================
# 2. Check input files
# ============================================================

print("=" * 70)
print("02. 1000 GENOMES EAS REFERENCE PREPARATION")
print("=" * 70)

for file in [
    GWAS_FILE,
    VCF_FILE,
    EAS_SAMPLE_FILE,
]:
    if not file.exists():
        raise FileNotFoundError(
            f"File not found: {file}"
        )


# ============================================================
# 3. Load GWAS SNP positions
# ============================================================

gwas = pd.read_csv(
    GWAS_FILE,
    sep="\t"
)

targets = gwas[
    [
        "chromosome",
        "base_pair_location",
    ]
].copy()

targets["CHROM"] = (
    "chr"
    + targets["chromosome"].astype(str)
)

targets["POS"] = (
    targets["base_pair_location"]
    .astype(int)
)

targets = targets[
    [
        "CHROM",
        "POS",
    ]
].drop_duplicates()

targets = targets.sort_values(
    [
        "CHROM",
        "POS",
    ]
)

print(
    f"GWAS target positions: "
    f"{len(targets):,}"
)


# ============================================================
# 4. Save bcftools target file
# ============================================================

targets.to_csv(
    TARGET_FILE,
    sep="\t",
    header=False,
    index=False
)

print(
    f"Target file: {TARGET_FILE}"
)


# ============================================================
# 5. Check EAS samples
# ============================================================

with open(
    EAS_SAMPLE_FILE,
    "r"
) as f:

    eas_samples = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(
    f"EAS samples: {len(eas_samples):,}"
)

if len(eas_samples) != 504:
    raise ValueError(
        f"Expected 504 EAS samples, "
        f"found {len(eas_samples)}."
    )


# ============================================================
# 6. Extract EAS target SNPs
# ============================================================

print()
print("Starting bcftools extraction...")
print("This step may take several hours because")
print("the source VCF is very large.")
print()

cmd = [
    "bcftools",
    "view",

    "-S",
    str(EAS_SAMPLE_FILE),

    "-T",
    str(TARGET_FILE),

    "-m2",
    "-M2",

    "-v",
    "snps",

    "-Oz",

    "-o",
    str(OUTPUT_VCF),

    str(VCF_FILE),
]

subprocess.run(
    cmd,
    check=True
)


# ============================================================
# 7. Index output
# ============================================================

print("\nIndexing output VCF...")

subprocess.run(
    [
        "bcftools",
        "index",
        "-t",
        str(OUTPUT_VCF),
    ],
    check=True
)


# ============================================================
# 8. Validate sample count
# ============================================================

result = subprocess.run(
    [
        "bcftools",
        "query",
        "-l",
        str(OUTPUT_VCF),
    ],
    capture_output=True,
    text=True,
    check=True,
)

output_samples = [
    x
    for x in result.stdout.splitlines()
    if x.strip()
]

print(
    f"Output samples: "
    f"{len(output_samples):,}"
)

if len(output_samples) != 504:
    raise ValueError(
        "Output VCF does not contain exactly "
        "504 EAS samples."
    )


# ============================================================
# 9. Validate variant count
# ============================================================

result = subprocess.run(
    [
        "bcftools",
        "index",
        "-n",
        str(OUTPUT_VCF),
    ],
    capture_output=True,
    text=True,
    check=True,
)

variant_count = int(
    result.stdout.strip()
)

print(
    f"Output variants: "
    f"{variant_count:,}"
)


# ============================================================
# 10. Final
# ============================================================

print()
print("=" * 70)
print("REFERENCE PREPARATION COMPLETE")
print("=" * 70)

print(
    f"EAS samples     : {len(output_samples):,}"
)

print(
    f"Target positions: {len(targets):,}"
)

print(
    f"Output variants : {variant_count:,}"
)

print(
    f"Output VCF      : {OUTPUT_VCF}"
)

print(
    f"Output index    : {OUTPUT_VCF}.tbi"
)