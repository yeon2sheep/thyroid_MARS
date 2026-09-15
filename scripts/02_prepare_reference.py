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
# 1번 코드에서 만든 GWAS 읽음

targets = gwas[
    [
        "chromosome",
        "base_pair_location",
    ]
].copy()
# 그 파일에서 염색체랑 위치만 가져옴 (22, 11012345)

targets["CHROM"] = (
    "chr"
    + targets["chromosome"].astype(str)
)
# 표기 맞춤 (22 -> chr22)

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
# 중복 제거..

targets = targets.sort_values(
    [
        "CHROM",
        "POS",
    ]
)
# 위치순으로 정렬 (결국 chr22 11012345 이런 식의 2열짜리 행렬 만들어짐)

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
# bcftools 용 타켓 파일 생성 (pandas로는 너무 많아서 못 읽음)

print(
    f"Target file: {TARGET_FILE}"
)


# ============================================================
# 5. Check EAS samples
# ============================================================

# EAS.sample.txt를 읽어서 sample ID 가져옴 HG0004 이런거
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
    str(EAS_SAMPLE_FILE), # 504명 번호들

    "-T",
    str(TARGET_FILE), 

    "-m2",
    "-M2", # allele 이 정확히 2개인 variant만

    "-v",
    "snps", # snp만 추출

    "-Oz", # 압축된 .vcf.gz 형태로 출력

    "-o",
    str(OUTPUT_VCF),

    str(VCF_FILE),
]
# bcftools 로 실제 추출
# chr22 전체 1000 지놈 VCF에서 EAS 504명 데이터 중 
# GWAS에서 사용되는 타겟 포지션에 해당하는 biallelic SNP만 뽑아서 
# 압축 VCF로 저장

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
# 인덱스 생성


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