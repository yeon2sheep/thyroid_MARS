import subprocess
from pathlib import Path
import pandas as pd


# =========================
# 1. Project paths
# =========================

PROJECT_DIR = Path(__file__).resolve().parents[1]

VCF_FILE = PROJECT_DIR / "data" / "reference" / "chr22.vcf"
EAS_SAMPLE_FILE = PROJECT_DIR / "data" / "reference" / "EAS.samples.txt"

MARS_SNP_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "mars_input"
    / "chr22_test_mars_snps.tsv"
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
    / "chr22_test_targets.tsv"
)

OUTPUT_VCF = (
    OUTPUT_DIR
    / "chr22_EAS504_testSNPs.vcf.gz"
)


# =========================
# 2. Check input files
# =========================

print("Checking input files...")

required_files = [
    VCF_FILE,
    EAS_SAMPLE_FILE,
    MARS_SNP_FILE,
]

for file in required_files:
    if not file.exists():
        raise FileNotFoundError(
            f"File not found: {file}"
        )

print("All input files found.")


# =========================
# 3. Load MARS SNP list
# =========================

print("\nLoading MARS SNP list...")

mars_snps = pd.read_csv(
    MARS_SNP_FILE,
    sep="\t"
)

print(
    f"Number of SNP records: "
    f"{len(mars_snps):,}"
)

required_columns = [
    "chromosome",
    "base_pair_location",
    "effect_allele_x",
    "other_allele_x",
]

for col in required_columns:
    if col not in mars_snps.columns:
        raise ValueError(
            f"Required column '{col}' not found.\n"
            f"Available columns: "
            f"{list(mars_snps.columns)}"
        )


# =========================
# 4. Create target position file
# =========================

print("\nCreating bcftools target file...")

targets = mars_snps[
    [
        "chromosome",
        "base_pair_location",
    ]
].copy()

targets.columns = [
    "CHROM",
    "POS",
]

targets["CHROM"] = (
    targets["CHROM"]
    .astype(str)
)

targets["POS"] = pd.to_numeric(
    targets["POS"],
    errors="coerce"
)

targets = targets.dropna()


# ---------------------------------
# IMPORTANT:
# The input VCF uses "chr22".
# Therefore, target chromosomes must
# also use "chr22".
# ---------------------------------

targets["CHROM"] = (
    targets["CHROM"]
    .replace(
        {
            "22": "chr22",
            "chr22": "chr22",
        }
    )
)


# Keep chromosome 22 only

targets = targets[
    targets["CHROM"] == "chr22"
]


# Convert position to integer

targets["POS"] = (
    targets["POS"]
    .astype(int)
)


# Remove duplicated positions

targets = targets.drop_duplicates()


# Sort by position

targets = targets.sort_values(
    ["CHROM", "POS"]
)


# Save target file

targets.to_csv(
    TARGET_FILE,
    sep="\t",
    index=False,
    header=False,
)

print(
    f"Target file: {TARGET_FILE}"
)

print(
    f"Unique target positions: "
    f"{len(targets):,}"
)


# =========================
# 5. Check EAS sample list
# =========================

print("\nChecking EAS sample list...")

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
    f"EAS samples: "
    f"{len(eas_samples)}"
)

if len(eas_samples) != 504:

    raise ValueError(
        f"Expected 504 EAS samples, "
        f"but found {len(eas_samples)}."
    )


# =========================
# 6. Extract EAS + target SNPs
# =========================

print(
    "\nStarting bcftools extraction..."
)

print(
    "Input VCF size is approximately 137 GB."
)

print(
    "bcftools will read the VCF "
    "and create a smaller output VCF."
)

print()

cmd = [
    "bcftools",
    "view",

    # Select EAS 504 samples
    "-S",
    str(EAS_SAMPLE_FILE),

    # Select target positions
    "-T",
    str(TARGET_FILE),

    # Keep biallelic variants
    "-m2",
    "-M2",

    # Keep SNPs only
    "-v",
    "snps",

    # Compress output
    "-Oz",

    "-o",
    str(OUTPUT_VCF),

    str(VCF_FILE),
]

print("Running bcftools...")
print()

subprocess.run(
    cmd,
    check=True
)


# =========================
# 7. Index output VCF
# =========================

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


# =========================
# 8. Verify output samples
# =========================

print("\nChecking output samples...")

result = subprocess.run(
    [
        "bcftools",
        "query",
        "-l",
        str(OUTPUT_VCF),
    ],
    capture_output=True,
    text=True,
    check=True
)

output_samples = [
    line.strip()
    for line in result.stdout.splitlines()
    if line.strip()
]

print(
    f"Output samples: "
    f"{len(output_samples)}"
)

if len(output_samples) != 504:

    raise ValueError(
        f"Expected 504 samples, "
        f"but found {len(output_samples)}."
    )


# =========================
# 9. Count output variants
# =========================

print("\nCounting output variants...")

result = subprocess.run(
    [
        "bcftools",
        "view",
        "-H",
        str(OUTPUT_VCF),
    ],
    capture_output=True,
    text=True,
    check=True
)

variant_count = len(
    result.stdout.splitlines()
)

print(
    f"Output variants: "
    f"{variant_count:,}"
)


# =========================
# 10. Final message
# =========================

print(
    "\n==================================="
)

print(
    "Reference preparation complete!"
)

print(
    "==================================="
)

print(
    f"EAS samples:       "
    f"{len(output_samples)}"
)

print(
    f"Target positions:  "
    f"{len(targets):,}"
)

print(
    f"Output variants:   "
    f"{variant_count:,}"
)

print("\nOutput VCF:")

print(
    OUTPUT_VCF
)

print("\nOutput index:")

print(
    str(OUTPUT_VCF) + ".tbi"
)