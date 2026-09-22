import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

FINAL_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "final"
)

GENE_BIN_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "gene_bin_inputs"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "gene_mars_inputs"
)

MEMBERSHIP_FILE = (
    GENE_BIN_DIR
    / "chr22_gene_bin_membership.tsv"
)

GENE_BIN_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "gene_bins"
    / "chr22_gene_bins_pm50.tsv"
)

Z_FILE = (
    FINAL_DIR
    / "chr22_final_z.tsv"
)

GENOTYPE_FILE = (
    FINAL_DIR
    / "chr22_final_genotype.tsv"
)


# ============================================================
# 2. Analysis settings
# ============================================================

MIN_SNPS_PER_BIN = 10
MAX_SNPS_PER_BIN = 1_000

CHROMOSOME = "22"


# ============================================================
# 3. Load gene-bin membership
# ============================================================

print("=" * 70)
print("07. PREPARE GENE-BASED MARS INPUTS")
print("=" * 70)

print()
print("Loading gene-bin membership...")

membership = pd.read_csv(
    MEMBERSHIP_FILE,
    sep="\t"
)

print(
    f"Gene-SNP assignments : "
    f"{len(membership):,}"
)

print(
    f"Gene bins             : "
    f"{membership['bin_id'].nunique():,}"
)


# ============================================================
# 4. Load gene-bin information
# ============================================================

print()
print("Loading gene-bin information...")

gene_bins = pd.read_csv(
    GENE_BIN_FILE,
    sep="\t"
)

print(
    f"Total gene bins       : "
    f"{len(gene_bins):,}"
)


# ============================================================
# 5. Load final Z scores
# ============================================================

print()
print("Loading final Z-scores...")

z = pd.read_csv(
    Z_FILE,
    sep="\t"
)

z["variant_key"] = (
    z["variant_key"]
    .astype(str)
)

print(
    f"Final Z-score records : "
    f"{len(z):,}"
)


# ============================================================
# 6. Load final genotype
# ============================================================

print()
print("Loading final genotype...")

genotype = pd.read_csv(
    GENOTYPE_FILE,
    sep="\t",
    index_col=0
)

genotype.index = (
    genotype.index
    .astype(str)
)

print(
    f"Final genotype        : "
    f"{genotype.shape[0]:,} SNPs × "
    f"{genotype.shape[1]:,} samples"
)


# ============================================================
# 7. Basic consistency checks
# ============================================================

print()
print("-" * 70)
print("CONSISTENCY CHECKS")
print("-" * 70)


if "bin_id" not in membership.columns:
    raise ValueError(
        "Membership file does not contain 'bin_id'."
    )


if "variant_key" not in membership.columns:
    raise ValueError(
        "Membership file does not contain 'variant_key'."
    )


required_gene_columns = {
    "bin_id",
    "gene_id",
    "gene_name",
    "bin_start",
    "bin_end",
}

missing_gene_columns = (
    required_gene_columns
    - set(gene_bins.columns)
)

if missing_gene_columns:
    raise ValueError(
        "Missing required columns in gene-bin file: "
        f"{missing_gene_columns}"
    )


# ------------------------------------------------------------
# Check duplicate gene-SNP assignments
# ------------------------------------------------------------

duplicate_assignments = (
    membership
    .duplicated(
        subset=["bin_id", "variant_key"]
    )
    .sum()
)

if duplicate_assignments > 0:
    raise ValueError(
        f"Duplicate gene-SNP assignments found: "
        f"{duplicate_assignments:,}"
    )


# ------------------------------------------------------------
# Check Z-score availability
# ------------------------------------------------------------

membership_keys = set(
    membership["variant_key"]
    .astype(str)
)

z_keys = set(
    z["variant_key"]
    .astype(str)
)

missing_z_keys = membership_keys - z_keys

if missing_z_keys:
    raise ValueError(
        f"{len(missing_z_keys):,} SNPs in gene bins "
        "do not have Z-scores."
    )


# ------------------------------------------------------------
# Check genotype availability
# ------------------------------------------------------------

genotype_keys = set(
    genotype.index
    .astype(str)
)

missing_genotype_keys = (
    membership_keys
    - genotype_keys
)

if missing_genotype_keys:
    raise ValueError(
        f"{len(missing_genotype_keys):,} SNPs in gene bins "
        "do not have genotype data."
    )


print("All consistency checks passed.")


# ============================================================
# 8. Restrict membership to chromosome 22
# ============================================================

membership["base_pair_location"] = pd.to_numeric(
    membership["base_pair_location"]
)

membership_chr = membership.copy()

print()
print("-" * 70)
print("GENE-BASED ANALYSIS DATA")
print("-" * 70)

print(
    f"Chromosome           : chr{CHROMOSOME}"
)

print(
    f"Gene-SNP assignments : "
    f"{len(membership_chr):,}"
)

print(
    f"Gene bins with SNPs  : "
    f"{membership_chr['bin_id'].nunique():,}"
)


# ============================================================
# 9. Create output directory
# ============================================================

if OUTPUT_DIR.exists():

    print()
    print(
        "Removing previous "
        "gene_mars_inputs directory..."
    )

    shutil.rmtree(
        OUTPUT_DIR
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 10. Count SNPs per gene bin
# ============================================================

print()
print("-" * 70)
print("CHECKING GENE BIN SIZES")
print("-" * 70)

bin_counts = (
    membership_chr
    .groupby("bin_id")
    .size()
    .sort_index()
)

print(
    f"Gene bins with SNPs : "
    f"{len(bin_counts):,}"
)

print(
    f"Smallest gene bin   : "
    f"{bin_counts.min():,} SNPs"
)

print(
    f"Largest gene bin    : "
    f"{bin_counts.max():,} SNPs"
)

large_bins = bin_counts[
    bin_counts > MAX_SNPS_PER_BIN
]

print(
    f"Bins > {MAX_SNPS_PER_BIN:,} SNPs : "
    f"{len(large_bins):,}"
)


# ============================================================
# 11. Remove bins with too few SNPs
# ============================================================

small_bins = bin_counts[
    bin_counts < MIN_SNPS_PER_BIN
].index.tolist()

if small_bins:

    print()
    print(
        f"Removing {len(small_bins):,} gene bins "
        f"with fewer than "
        f"{MIN_SNPS_PER_BIN} SNPs."
    )

    membership_chr = membership_chr[
        ~membership_chr["bin_id"].isin(
            small_bins
        )
    ].copy()

else:

    print()
    print(
        f"No gene bins have fewer than "
        f"{MIN_SNPS_PER_BIN} SNPs."
    )


# ============================================================
# 12. Split oversized gene bins if necessary
# ============================================================

print()
print("-" * 70)
print("PREPARING ANALYSIS BINS")
print("-" * 70)

final_bin_records = []

for gene_bin_id, group in membership_chr.groupby(
    "bin_id",
    sort=False
):

    group = (
        group
        .sort_values(
            "base_pair_location"
        )
        .reset_index(drop=True)
    )

    n = len(group)

    # --------------------------------------------------------
    # Normal gene bin
    # --------------------------------------------------------

    if n <= MAX_SNPS_PER_BIN:

        group["analysis_bin_id"] = (
            gene_bin_id
        )

        final_bin_records.append(
            group
        )

    # --------------------------------------------------------
    # Oversized gene bin
    # --------------------------------------------------------

    else:

        print(
            f"Splitting large gene bin "
            f"{gene_bin_id}: "
            f"{n:,} SNPs"
        )

        n_subbins = int(
            np.ceil(
                n / MAX_SNPS_PER_BIN
            )
        )

        for sub_idx in range(
            n_subbins
        ):

            start_idx = (
                sub_idx
                * MAX_SNPS_PER_BIN
            )

            end_idx = min(
                (sub_idx + 1)
                * MAX_SNPS_PER_BIN,
                n
            )

            sub_group = (
                group
                .iloc[
                    start_idx:end_idx
                ]
                .copy()
            )

            sub_group[
                "analysis_bin_id"
            ] = (
                f"{gene_bin_id}"
                f"_part{sub_idx + 1}"
            )

            final_bin_records.append(
                sub_group
            )


if not final_bin_records:

    raise ValueError(
        "No analysis bins were created."
    )


membership_binned = pd.concat(
    final_bin_records,
    ignore_index=True
)


# ============================================================
# 13. Create gene-bin summary
# ============================================================

bin_summary = (
    membership_binned
    .groupby(
        "analysis_bin_id"
    )
    .agg(
        gene_bin_id=(
            "bin_id",
            "first"
        ),
        gene_id=(
            "gene_id",
            "first"
        ),
        gene_name=(
            "gene_name",
            "first"
        ),
        bin_start=(
            "bin_start",
            "first"
        ),
        bin_end=(
            "bin_end",
            "first"
        ),
        start_position=(
            "base_pair_location",
            "min"
        ),
        end_position=(
            "base_pair_location",
            "max"
        ),
        n_snps=(
            "variant_key",
            "count"
        ),
    )
    .reset_index()
)


bin_summary = (
    bin_summary
    .sort_values(
        [
            "start_position",
            "end_position"
        ]
    )
    .reset_index(drop=True)
)


bin_summary.insert(
    0,
    "bin_number",
    np.arange(
        1,
        len(bin_summary) + 1
    )
)


print()
print(
    f"Analysis bins created : "
    f"{len(bin_summary):,}"
)

print(
    f"Smallest bin         : "
    f"{bin_summary['n_snps'].min():,} SNPs"
)

print(
    f"Largest bin          : "
    f"{bin_summary['n_snps'].max():,} SNPs"
)


# ============================================================
# 14. Save bin summary
# ============================================================

bin_summary_file = (
    OUTPUT_DIR
    / "chr22_gene_bin_summary.tsv"
)

bin_summary.to_csv(
    bin_summary_file,
    sep="\t",
    index=False
)


# ============================================================
# 15. Prepare each gene bin
# ============================================================

print()
print("=" * 70)
print("PREPARING GENE-BIN MARS INPUTS")
print("=" * 70)

summary_records = []


# ------------------------------------------------------------
# Create Z lookup
# ------------------------------------------------------------

z_lookup = (
    z
    .set_index("variant_key")
)


for _, bin_row in bin_summary.iterrows():

    bin_number = int(
        bin_row["bin_number"]
    )

    analysis_bin_id = (
        bin_row["analysis_bin_id"]
    )

    gene_bin_id = (
        bin_row["gene_bin_id"]
    )

    print()
    print(
        f"[{bin_number}/{len(bin_summary)}] "
        f"{analysis_bin_id}"
    )

    bin_dir = (
        OUTPUT_DIR
        / f"bin_{bin_number:03d}"
    )

    bin_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # Select SNPs
    # ========================================================

    bin_snps = membership_binned[
        membership_binned[
            "analysis_bin_id"
        ]
        == analysis_bin_id
    ].copy()

    bin_snps = (
        bin_snps
        .sort_values(
            "base_pair_location"
        )
        .reset_index(drop=True)
    )

    bin_keys = (
        bin_snps[
            "variant_key"
        ]
        .astype(str)
        .tolist()
    )


    # ========================================================
    # Z-score
    # ========================================================

    bin_z = (
        z_lookup
        .loc[bin_keys]
        .reset_index()
    )


    # ========================================================
    # Genotype
    # ========================================================

    bin_genotype = (
        genotype
        .loc[bin_keys]
        .copy()
    )


    # ========================================================
    # Verify dimensions
    # ========================================================

    n_snps = len(
        bin_keys
    )

    n_samples = (
        bin_genotype.shape[1]
    )


    if n_snps != bin_z.shape[0]:

        raise ValueError(
            f"{analysis_bin_id}: "
            "SNP/Z mismatch."
        )


    if (
        n_snps
        != bin_genotype.shape[0]
    ):

        raise ValueError(
            f"{analysis_bin_id}: "
            "SNP/genotype mismatch."
        )


    # ========================================================
    # Check genotype values
    # ========================================================

    genotype_values = (
        bin_genotype
        .to_numpy(
            dtype=float
        )
    )


    if not np.isfinite(
        genotype_values
    ).all():

        raise ValueError(
            f"{analysis_bin_id}: "
            "genotype contains "
            "non-finite values."
        )


    if (
        genotype_values.min()
        < 0
        or
        genotype_values.max()
        > 2
    ):

        raise ValueError(
            f"{analysis_bin_id}: "
            "genotype values "
            "outside [0, 2]."
        )


    # ========================================================
    # Calculate LD matrix
    # ========================================================

    print(
        f"  Gene       : "
        f"{bin_snps['gene_id'].iloc[0]}"
    )

    print(
        f"  SNPs       : "
        f"{n_snps:,}"
    )

    print(
        "  Calculating LD matrix..."
    )


    ld = np.corrcoef(
        genotype_values
    )


    ld = np.asarray(
        ld,
        dtype=np.float64
    )


    # Numerical cleanup only

    ld[
        np.abs(ld) < 1e-12
    ] = 0.0

    np.fill_diagonal(
        ld,
        1.0
    )


    if not np.isfinite(
        ld
    ).all():

        raise ValueError(
            f"{analysis_bin_id}: "
            "LD matrix contains "
            "non-finite values."
        )


    # ========================================================
    # Save SNP metadata
    # ========================================================

    bin_snps_file = (
        bin_dir
        / "snps.tsv"
    )


    bin_snps[
    [
        "variant_key",
        "base_pair_location",
        "gene_id",
        "gene_name",
        "bin_start",
        "bin_end",
        ]
    ].to_csv(
        bin_snps_file,
        sep="\t",
        index=False
    )


    # ========================================================
    # Save Z scores
    # ========================================================

    bin_z_file = (
        bin_dir
        / "z.tsv"
    )


    bin_z[
        [
            "variant_key",
            "z"
        ]
    ].to_csv(
        bin_z_file,
        sep="\t",
        index=False
    )


    # ========================================================
    # Save genotype
    # ========================================================

    bin_genotype_file = (
        bin_dir
        / "genotype.tsv"
    )


    bin_genotype.to_csv(
        bin_genotype_file,
        sep="\t"
    )


    # ========================================================
    # Save LD matrix
    # ========================================================

    ld_file = (
        bin_dir
        / "ld.tsv"
    )


    ld_df = pd.DataFrame(
        ld,
        index=bin_keys,
        columns=bin_keys
    )


    ld_df.to_csv(
        ld_file,
        sep="\t"
    )


    # ========================================================
    # Record summary
    # ========================================================

    summary_records.append(
        {
            "bin_number": bin_number,
            "analysis_bin_id": analysis_bin_id,
            "gene_bin_id": gene_bin_id,
            "gene_id": bin_snps[
                "gene_id"
            ].iloc[0],
            "gene_name": bin_snps[
                "gene_name"
            ].iloc[0],
            "bin_start": int(
                bin_snps[
                    "bin_start"
                ].iloc[0]
            ),
            "bin_end": int(
                bin_snps[
                    "bin_end"
                ].iloc[0]
            ),
            "start_position": int(
                bin_snps[
                    "base_pair_location"
                ].min()
            ),
            "end_position": int(
                bin_snps[
                    "base_pair_location"
                ].max()
            ),
            "n_snps": n_snps,
            "n_samples": n_samples,
            "ld_shape": (
                f"{n_snps}x{n_snps}"
            ),
            "snps_file": str(
                bin_snps_file
                .relative_to(
                    PROJECT_DIR
                )
            ),
            "z_file": str(
                bin_z_file
                .relative_to(
                    PROJECT_DIR
                )
            ),
            "genotype_file": str(
                bin_genotype_file
                .relative_to(
                    PROJECT_DIR
                )
            ),
            "ld_file": str(
                ld_file
                .relative_to(
                    PROJECT_DIR
                )
            ),
        }
    )


    print(
        f"  LD shape   : "
        f"{n_snps:,} × {n_snps:,}"
    )

    print(
        "  Complete."
    )


# ============================================================
# 16. Save final summary
# ============================================================

final_summary = pd.DataFrame(
    summary_records
)


final_summary_file = (
    OUTPUT_DIR
    / "chr22_gene_mars_input_summary.tsv"
)


final_summary.to_csv(
    final_summary_file,
    sep="\t",
    index=False
)


# ============================================================
# 17. Final checks
# ============================================================

total_binned_snps = int(
    final_summary[
        "n_snps"
    ].sum()
)


if (
    total_binned_snps
    != len(membership_binned)
):

    raise ValueError(
        "Final bin SNP count mismatch."
    )


print()
print("=" * 70)
print("07. GENE-BASED MARS INPUT PREPARATION COMPLETE")
print("=" * 70)

print(
    f"Gene bins with SNPs     : "
    f"{membership_chr['bin_id'].nunique():,}"
)

print(
    f"Analysis bins           : "
    f"{len(final_summary):,}"
)

print(
    f"SNPs assigned to bins   : "
    f"{total_binned_snps:,}"
)

print(
    f"Output directory        : "
    f"{OUTPUT_DIR}"
)

print(
    f"Gene-bin summary        : "
    f"{bin_summary_file}"
)

print(
    f"MARS input summary      : "
    f"{final_summary_file}"
)

print()
print(
    "MARS execution is NOT performed."
)

print(
    "The generated files are prepared "
    "for the next MARS execution step."
)

print("=" * 70)