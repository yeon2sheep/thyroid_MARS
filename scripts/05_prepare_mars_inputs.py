import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

FINAL_DIR = PROJECT_DIR / "data" / "processed" / "final"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "mars_bin_inputs_v2"

SNPS_FILE = FINAL_DIR / "chr22_final_snps.tsv"
Z_FILE = FINAL_DIR / "chr22_final_z.tsv"
GENOTYPE_FILE = FINAL_DIR / "chr22_final_genotype.tsv"


# ============================================================
# 2. Analysis settings
# ============================================================

BIN_SIZE = 500_000

MIN_SNPS_PER_BIN = 10
MAX_SNPS_PER_BIN = 1_000

CHROMOSOME = "chr22"


# ============================================================
# 3. Load final QC data
# ============================================================

print("=" * 70)
print("05. PREPARE MARS INPUTS")
print("=" * 70)

print()
print("Loading final QC data...")

snps = pd.read_csv(SNPS_FILE, sep="\t")
z = pd.read_csv(Z_FILE, sep="\t")
genotype = pd.read_csv(GENOTYPE_FILE, sep="\t", index_col=0)

print(f"Final SNP metadata : {len(snps):,}")
print(f"Final Z-scores     : {len(z):,}")
print(
    f"Final genotype     : "
    f"{genotype.shape[0]:,} SNPs × {genotype.shape[1]:,} samples"
)


# ============================================================
# 4. Basic consistency checks
# ============================================================

required_snp_columns = {
    "chromosome",
    "base_pair_location",
    "variant_key",
}

missing_columns = required_snp_columns - set(snps.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns in SNP file: {missing_columns}"
    )


if len(snps) != len(z):
    raise ValueError(
        f"SNP/Z length mismatch: {len(snps)} vs {len(z)}"
    )


if len(snps) != genotype.shape[0]:
    raise ValueError(
        f"SNP/genotype length mismatch: "
        f"{len(snps)} vs {genotype.shape[0]}"
    )


# ------------------------------------------------------------
# Make sure all three datasets have the same SNP order
# ------------------------------------------------------------

snp_keys = snps["variant_key"].astype(str).tolist()
z_keys = z["variant_key"].astype(str).tolist()
genotype_keys = genotype.index.astype(str).tolist()

if snp_keys != z_keys:
    raise ValueError("SNP order does not match between SNP and Z files.")

if snp_keys != genotype_keys:
    raise ValueError(
        "SNP order does not match between SNP and genotype files."
    )


# ============================================================
# 5. Restrict to chromosome
# ============================================================

snps["chromosome"] = snps["chromosome"].astype(str)

snps_chr = snps[
    snps["chromosome"].isin([CHROMOSOME, CHROMOSOME.replace("chr", "")])
].copy()

if len(snps_chr) == 0:
    raise ValueError(
        f"No SNPs found for chromosome {CHROMOSOME}."
    )

snps_chr = snps_chr.sort_values(
    "base_pair_location"
).reset_index(drop=True)


# Reorder Z and genotype using the SNP order after sorting
ordered_keys = snps_chr["variant_key"].astype(str).tolist()

z_lookup = z.copy()
z_lookup["variant_key"] = z_lookup["variant_key"].astype(str)

z_chr = (
    z_lookup
    .set_index("variant_key")
    .loc[ordered_keys]
    .reset_index()
)

genotype_chr = genotype.loc[ordered_keys].copy()


print()
print("-" * 70)
print("CHROMOSOME")
print("-" * 70)

print(f"Chromosome       : {CHROMOSOME}")
print(f"Analysis SNPs    : {len(snps_chr):,}")
print(
    f"Position range   : "
    f"{snps_chr['base_pair_location'].min():,} - "
    f"{snps_chr['base_pair_location'].max():,}"
)


# ============================================================
# 6. Create output directory
# ============================================================

if OUTPUT_DIR.exists():
    print()
    print("Removing previous mars_bin_inputs_v2 directory...")
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 7. Assign genomic bins
# ============================================================

print()
print("-" * 70)
print("CREATING ANALYSIS BINS")
print("-" * 70)

snps_chr["bin_start"] = (
    (snps_chr["base_pair_location"] - 1)
    // BIN_SIZE
) * BIN_SIZE + 1

snps_chr["bin_end"] = (
    snps_chr["bin_start"] + BIN_SIZE - 1
)

snps_chr["bin_id"] = (
    CHROMOSOME
    + "_"
    + snps_chr["bin_start"].astype(str)
    + "_"
    + snps_chr["bin_end"].astype(str)
)


# ============================================================
# 8. Split oversized bins if necessary
# ============================================================

final_bin_records = []

for original_bin_id, group in snps_chr.groupby(
    "bin_id",
    sort=False
):

    group = group.sort_values(
        "base_pair_location"
    ).reset_index(drop=True)

    n = len(group)

    if n <= MAX_SNPS_PER_BIN:

        group["analysis_bin_id"] = original_bin_id

        final_bin_records.append(group)

    else:

        print(
            f"Splitting large bin {original_bin_id}: "
            f"{n:,} SNPs"
        )

        # Number of sub-bins required
        n_subbins = int(
            np.ceil(n / MAX_SNPS_PER_BIN)
        )

        # Use pandas iloc instead of np.array_split
        # so each object remains a DataFrame.
        for sub_idx in range(n_subbins):

            start_idx = (
                sub_idx * MAX_SNPS_PER_BIN
            )

            end_idx = min(
                (sub_idx + 1) * MAX_SNPS_PER_BIN,
                n
            )

            sub_group = group.iloc[
                start_idx:end_idx
            ].copy()

            start_pos = int(
                sub_group[
                    "base_pair_location"
                ].min()
            )

            end_pos = int(
                sub_group[
                    "base_pair_location"
                ].max()
            )

            sub_group["analysis_bin_id"] = (
                f"{original_bin_id}_part{sub_idx + 1}"
            )

            final_bin_records.append(
                sub_group
            )


if not final_bin_records:
    raise ValueError(
        "No analysis bins were created."
    )


snps_binned = pd.concat(
    final_bin_records,
    ignore_index=True
)


# ============================================================
# 9. Remove bins with too few SNPs
# ============================================================

bin_counts = (
    snps_binned
    .groupby("analysis_bin_id")
    .size()
    .sort_index()
)

small_bins = bin_counts[
    bin_counts < MIN_SNPS_PER_BIN
].index.tolist()

if small_bins:
    print()
    print(
        f"Removing {len(small_bins)} bins "
        f"with fewer than {MIN_SNPS_PER_BIN} SNPs."
    )

    snps_binned = snps_binned[
        ~snps_binned["analysis_bin_id"].isin(
            small_bins
        )
    ].copy()


# ============================================================
# 10. Create bin summary
# ============================================================

bin_summary = (
    snps_binned
    .groupby("analysis_bin_id")
    .agg(
        chromosome=("chromosome", "first"),
        start_position=("base_pair_location", "min"),
        end_position=("base_pair_location", "max"),
        n_snps=("variant_key", "count"),
    )
    .reset_index()
)


bin_summary = bin_summary.sort_values(
    "start_position"
).reset_index(drop=True)

bin_summary.insert(
    0,
    "bin_number",
    np.arange(1, len(bin_summary) + 1)
)


print()
print(f"Analysis bins created : {len(bin_summary):,}")
print(
    f"Smallest bin         : "
    f"{bin_summary['n_snps'].min():,} SNPs"
)
print(
    f"Largest bin          : "
    f"{bin_summary['n_snps'].max():,} SNPs"
)


# ============================================================
# 11. Save bin summary
# ============================================================

bin_summary_file = (
    OUTPUT_DIR / "chr22_bin_summary.tsv"
)

bin_summary.to_csv(
    bin_summary_file,
    sep="\t",
    index=False
)


# ============================================================
# 12. Prepare each bin
# ============================================================

print()
print("=" * 70)
print("PREPARING BIN-WISE MARS INPUTS")
print("=" * 70)

summary_records = []


for _, bin_row in bin_summary.iterrows():

    bin_number = int(
        bin_row["bin_number"]
    )

    bin_id = bin_row["analysis_bin_id"]

    print()
    print(
        f"[{bin_number}/{len(bin_summary)}] "
        f"{bin_id}"
    )

    bin_dir = OUTPUT_DIR / f"bin_{bin_number:03d}"

    bin_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Select SNPs
    # --------------------------------------------------------

    bin_snps = snps_binned[
        snps_binned["analysis_bin_id"] == bin_id
    ].copy()

    bin_snps = bin_snps.sort_values(
        "base_pair_location"
    ).reset_index(drop=True)

    bin_keys = (
        bin_snps["variant_key"]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # Z-score
    # --------------------------------------------------------

    bin_z = (
        z_chr
        .set_index("variant_key")
        .loc[bin_keys]
        .reset_index()
    )

    # --------------------------------------------------------
    # Genotype
    # --------------------------------------------------------

    bin_genotype = (
        genotype_chr
        .loc[bin_keys]
        .copy()
    )

    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    n_snps = len(bin_keys)
    n_samples = bin_genotype.shape[1]

    if n_snps != bin_z.shape[0]:
        raise ValueError(
            f"{bin_id}: SNP/Z mismatch."
        )

    if n_snps != bin_genotype.shape[0]:
        raise ValueError(
            f"{bin_id}: SNP/genotype mismatch."
        )

    # --------------------------------------------------------
    # Check genotype values
    # --------------------------------------------------------

    genotype_values = (
        bin_genotype.to_numpy(dtype=float)
    )

    if not np.isfinite(
        genotype_values
    ).all():
        raise ValueError(
            f"{bin_id}: genotype contains "
            "non-finite values."
        )

    if (
        genotype_values.min() < 0
        or genotype_values.max() > 2
    ):
        raise ValueError(
            f"{bin_id}: genotype values "
            "outside [0, 2]."
        )

    # --------------------------------------------------------
    # Calculate LD matrix
    #
    # Pearson correlation of genotype dosages
    # --------------------------------------------------------

    print(
        f"  SNPs       : {n_snps:,}"
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
    ld[np.abs(ld) < 1e-12] = 0.0
    np.fill_diagonal(ld, 1.0)

    if not np.isfinite(ld).all():
        raise ValueError(
            f"{bin_id}: LD matrix contains "
            "non-finite values."
        )

    # --------------------------------------------------------
    # Save SNP metadata
    # --------------------------------------------------------

    bin_snps_file = (
        bin_dir / "snps.tsv"
    )

    bin_snps[
        [
            "variant_key",
            "chromosome",
            "base_pair_location",
            "effect_allele",
            "other_allele",
            "z",
            "analysis_bin_id",
        ]
    ].to_csv(
        bin_snps_file,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Save Z scores
    # --------------------------------------------------------

    bin_z_file = (
        bin_dir / "z.tsv"
    )

    bin_z[
        ["variant_key", "z"]
    ].to_csv(
        bin_z_file,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Save genotype
    # --------------------------------------------------------

    bin_genotype_file = (
        bin_dir / "genotype.tsv"
    )

    bin_genotype.to_csv(
        bin_genotype_file,
        sep="\t"
    )

    # --------------------------------------------------------
    # Save LD matrix
    #
    # Rows/columns use variant_key so that
    # SNP order is explicit.
    # --------------------------------------------------------

    ld_file = (
        bin_dir / "ld.tsv"
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

    # --------------------------------------------------------
    # Record summary
    # --------------------------------------------------------

    summary_records.append(
        {
            "bin_number": bin_number,
            "bin_id": bin_id,
            "start_position": int(
                bin_row["start_position"]
            ),
            "end_position": int(
                bin_row["end_position"]
            ),
            "n_snps": n_snps,
            "n_samples": n_samples,
            "ld_shape": f"{n_snps}x{n_snps}",
            "snps_file": str(
                bin_snps_file.relative_to(PROJECT_DIR)
            ),
            "z_file": str(
                bin_z_file.relative_to(PROJECT_DIR)
            ),
            "genotype_file": str(
                bin_genotype_file.relative_to(PROJECT_DIR)
            ),
            "ld_file": str(
                ld_file.relative_to(PROJECT_DIR)
            ),
        }
    )

    print(
        f"  LD shape   : {n_snps:,} × {n_snps:,}"
    )

    print(
        "  Complete."
    )


# ============================================================
# 13. Save final summary
# ============================================================

final_summary = pd.DataFrame(
    summary_records
)

final_summary_file = (
    OUTPUT_DIR / "chr22_mars_input_summary.tsv"
)

final_summary.to_csv(
    final_summary_file,
    sep="\t",
    index=False
)


# ============================================================
# 14. Final checks
# ============================================================

total_binned_snps = int(
    final_summary["n_snps"].sum()
)

if total_binned_snps != len(snps_binned):
    raise ValueError(
        "Final bin SNP count mismatch."
    )


print()
print("=" * 70)
print("05. MARS INPUT PREPARATION COMPLETE")
print("=" * 70)

print(
    f"Final QC SNPs       : {len(snps_chr):,}"
)

print(
    f"SNPs assigned to bins: "
    f"{total_binned_snps:,}"
)

print(
    f"Analysis bins       : "
    f"{len(final_summary):,}"
)

print(
    f"Output directory    : "
    f"{OUTPUT_DIR}"
)

print(
    f"Bin summary         : "
    f"{bin_summary_file}"
)

print(
    f"MARS summary        : "
    f"{final_summary_file}"
)

print()
print("MARS execution is NOT performed.")
print(
    "The generated files are prepared for "
    "the next MARS execution step."
)
print("=" * 70)