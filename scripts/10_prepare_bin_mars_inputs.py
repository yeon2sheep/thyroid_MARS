from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

BIN_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "bins"
    / "chr22_test_bin_membership.tsv"
)

GENOTYPE_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "mars_input"
    / "chr22_EAS504_genotype_final.tsv"
)

Z_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "mars_input"
    / "chr22_gwas_z_final.tsv"
)

SNP_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "mars_input"
    / "chr22_final_snps.tsv"
)

LD_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "mars_input"
    / "chr22_LD_final.tsv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "mars_bin_inputs"
)


# ============================================================
# 2. Create output directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. Load bin membership
# ============================================================

print("=" * 60)
print("Loading bin membership")
print("=" * 60)

bins = pd.read_csv(
    BIN_FILE,
    sep="\t"
)

print(f"Bin membership records : {len(bins):,}")
print(f"Columns                : {list(bins.columns)}")

required_bin_columns = {
    "bin_id",
    "variant_key"
}

missing_columns = (
    required_bin_columns
    - set(bins.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing columns in bin membership: "
        f"{missing_columns}"
    )


# ============================================================
# 4. Load final SNP list
# ============================================================

print("\n" + "=" * 60)
print("Loading final SNP list")
print("=" * 60)

snps = pd.read_csv(
    SNP_FILE,
    sep="\t"
)

if "variant_key" not in snps.columns:
    raise ValueError(
        "variant_key column not found in final SNP file."
    )

final_variant_keys = (
    snps["variant_key"]
    .astype(str)
    .tolist()
)

print(
    f"Final SNPs : "
    f"{len(final_variant_keys):,}"
)

if len(final_variant_keys) != len(
    set(final_variant_keys)
):
    raise ValueError(
        "Duplicate variant_key detected in final SNP list."
    )

print("✓ Final SNP list contains no duplicates")


# ============================================================
# 5. Load genotype
# ============================================================

print("\n" + "=" * 60)
print("Loading final genotype")
print("=" * 60)

genotype = pd.read_csv(
    GENOTYPE_FILE,
    sep="\t"
)

if "variant_key" not in genotype.columns:
    raise ValueError(
        "variant_key column not found in genotype."
    )

geno_variant_keys = (
    genotype["variant_key"]
    .astype(str)
    .tolist()
)

sample_columns = list(
    genotype.columns[1:]
)

n_samples = len(sample_columns)

print(
    f"Genotype SNPs : "
    f"{len(geno_variant_keys):,}"
)

print(
    f"EAS samples   : "
    f"{n_samples:,}"
)

if geno_variant_keys != final_variant_keys:
    raise ValueError(
        "Genotype SNP order does not match final SNP list."
    )

print("✓ Genotype SNP order matches final SNP list")


# ============================================================
# 6. Load Z-scores
# ============================================================

print("\n" + "=" * 60)
print("Loading final Z-scores")
print("=" * 60)

z_df = pd.read_csv(
    Z_FILE,
    sep="\t"
)

if "variant_key" not in z_df.columns:
    raise ValueError(
        "variant_key column not found in Z-score file."
    )

z_variant_keys = (
    z_df["variant_key"]
    .astype(str)
    .tolist()
)

if z_variant_keys != final_variant_keys:
    raise ValueError(
        "Z-score SNP order does not match final SNP list."
    )

print(
    f"Z-score SNPs : "
    f"{len(z_variant_keys):,}"
)

print("✓ Z-score SNP order matches final SNP list")


# ============================================================
# 7. Extract Z-score column
# ============================================================

if "z" not in z_df.columns:
    raise ValueError(
        "z column not found in Z-score file."
    )

z_values = z_df["z"].to_numpy(
    dtype=np.float64
)

if np.isnan(z_values).any():
    raise ValueError(
        "Z-score contains NaN."
    )

if np.isinf(z_values).any():
    raise ValueError(
        "Z-score contains Inf."
    )

print("✓ Z-scores contain no NaN/Inf")


# ============================================================
# 8. Load LD matrix
# ============================================================

print("\n" + "=" * 60)
print("Loading LD matrix")
print("=" * 60)

ld = pd.read_csv(
    LD_FILE,
    sep="\t",
    header=None
).to_numpy(
    dtype=np.float64
)

print(
    f"LD matrix : "
    f"{ld.shape[0]:,} × {ld.shape[1]:,}"
)

expected_shape = (
    len(final_variant_keys),
    len(final_variant_keys)
)

if ld.shape != expected_shape:
    raise ValueError(
        f"LD shape {ld.shape} does not match "
        f"expected {expected_shape}."
    )

print("✓ LD dimensions match final SNP set")


# ============================================================
# 9. Create lookup table
# ============================================================

variant_to_index = {
    variant_key: index
    for index, variant_key
    in enumerate(final_variant_keys)
}


# ============================================================
# 10. Remove bins with missing final SNPs
# ============================================================

print("\n" + "=" * 60)
print("Checking bin membership")
print("=" * 60)

bins["variant_key"] = (
    bins["variant_key"]
    .astype(str)
)

missing_from_final = (
    ~bins["variant_key"].isin(
        variant_to_index
    )
)

missing_count = missing_from_final.sum()

print(
    f"Bin SNPs not present in final set : "
    f"{missing_count:,}"
)

if missing_count > 0:
    print(
        "These SNPs will be excluded because "
        "they were removed during QC."
    )

bins = bins[
    ~missing_from_final
].copy()

print(
    f"Usable bin membership records : "
    f"{len(bins):,}"
)


# ============================================================
# 11. Prepare each bin
# ============================================================

print("\n" + "=" * 60)
print("Preparing bin-level MARS inputs")
print("=" * 60)

bin_ids = (
    bins["bin_id"]
    .drop_duplicates()
    .tolist()
)

print(
    f"Number of bins : "
    f"{len(bin_ids)}"
)

summary_rows = []


for bin_number, bin_id in enumerate(
    bin_ids,
    start=1
):

    print("\n" + "-" * 60)
    print(
        f"BIN {bin_number:02d} / "
        f"{len(bin_ids):02d}"
    )
    print(
        f"Bin ID : {bin_id}"
    )

    # --------------------------------------------------------
    # Get SNPs belonging to this bin
    # --------------------------------------------------------

    bin_members = bins[
        bins["bin_id"] == bin_id
    ].copy()

    bin_variant_keys = (
        bin_members["variant_key"]
        .astype(str)
        .tolist()
    )

    # Remove accidental duplicates
    bin_variant_keys = list(
        dict.fromkeys(
            bin_variant_keys
        )
    )

    m = len(bin_variant_keys)

    print(
        f"SNPs in bin : {m:,}"
    )

    if m == 0:
        print("⚠ Empty bin. Skipping.")
        continue

    # --------------------------------------------------------
    # Convert variant keys to global indices
    # --------------------------------------------------------

    indices = np.array(
        [
            variant_to_index[v]
            for v in bin_variant_keys
        ],
        dtype=int
    )

    # --------------------------------------------------------
    # Create bin directory
    # --------------------------------------------------------

    safe_bin_id = (
        str(bin_id)
        .replace("/", "_")
        .replace("\\", "_")
    )

    bin_dir = (
        OUTPUT_DIR
        / safe_bin_id
    )

    bin_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. SNP information
    # --------------------------------------------------------

    bin_snp_df = snps[
        snps["variant_key"].isin(
            bin_variant_keys
        )
    ].copy()

    # Reorder exactly according to bin_variant_keys
    bin_snp_df["__order"] = (
        bin_snp_df["variant_key"]
        .map(
            {
                v: i
                for i, v
                in enumerate(bin_variant_keys)
            }
        )
    )

    bin_snp_df = (
        bin_snp_df
        .sort_values("__order")
        .drop(columns="__order")
    )

    if (
        bin_snp_df["variant_key"].tolist()
        != bin_variant_keys
    ):
        raise ValueError(
            f"SNP order mismatch in {bin_id}"
        )

    bin_snp_df.to_csv(
        bin_dir / "snps.tsv",
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # 2. Z-score
    # --------------------------------------------------------

    bin_z = z_values[
        indices
    ]

    if len(bin_z) != m:
        raise ValueError(
            f"Z-score length mismatch in {bin_id}"
        )

    z_output = pd.DataFrame(
        {
            "variant_key": bin_variant_keys,
            "z": bin_z
        }
    )

    z_output.to_csv(
        bin_dir / "z.tsv",
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # 3. Genotype
    # --------------------------------------------------------

    bin_genotype = genotype[
        genotype["variant_key"].isin(
            bin_variant_keys
        )
    ].copy()

    bin_genotype["__order"] = (
        bin_genotype["variant_key"]
        .map(
            {
                v: i
                for i, v
                in enumerate(bin_variant_keys)
            }
        )
    )

    bin_genotype = (
        bin_genotype
        .sort_values("__order")
        .drop(columns="__order")
    )

    if (
        bin_genotype["variant_key"].tolist()
        != bin_variant_keys
    ):
        raise ValueError(
            f"Genotype order mismatch in {bin_id}"
        )

    bin_genotype.to_csv(
        bin_dir / "genotype.tsv",
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # 4. LD matrix
    # --------------------------------------------------------

    bin_ld = ld[
        np.ix_(
            indices,
            indices
        )
    ]

    if bin_ld.shape != (
        m,
        m
    ):
        raise ValueError(
            f"LD shape mismatch in {bin_id}: "
            f"{bin_ld.shape}"
        )

    # Symmetry check
    symmetry_error = np.max(
        np.abs(
            bin_ld
            - bin_ld.T
        )
    )

    if symmetry_error > 1e-8:
        raise ValueError(
            f"LD matrix is not symmetric "
            f"in {bin_id}"
        )

    # Diagonal check
    diagonal_error = np.max(
        np.abs(
            np.diag(bin_ld)
            - 1.0
        )
    )

    if diagonal_error > 1e-8:
        raise ValueError(
            f"LD diagonal is not 1 "
            f"in {bin_id}"
        )

    pd.DataFrame(
        bin_ld
    ).to_csv(
        bin_dir / "ld.tsv",
        sep="\t",
        header=False,
        index=False
    )

    # --------------------------------------------------------
    # 5. Create MARS-compatible genotype file
    # --------------------------------------------------------
    #
    # MARS expects:
    #
    # SNP_ID sample1 sample2 ... sampleN
    #
    # No header.
    #

    mars_genotype = bin_genotype.copy()

    mars_genotype.to_csv(
        bin_dir / "mars_genotype.txt",
        sep=" ",
        index=False,
        header=False
    )

    # --------------------------------------------------------
    # 6. Create MARS-compatible Z file
    # --------------------------------------------------------
    #
    # MARS expects a simple whitespace-separated
    # vector of Z-scores.
    #

    np.savetxt(
        bin_dir / "mars_z.txt",
        bin_z.reshape(1, -1),
        fmt="%.10g"
    )

    # --------------------------------------------------------
    # 7. Summary
    # --------------------------------------------------------

    summary_rows.append(
        {
            "bin_id": bin_id,
            "n_snps": m,
            "n_samples": n_samples,
            "genotype_rows": len(
                mars_genotype
            ),
            "z_scores": len(
                bin_z
            ),
            "ld_rows": bin_ld.shape[0],
            "ld_cols": bin_ld.shape[1],
            "ld_symmetry_error":
                symmetry_error,
            "ld_diagonal_error":
                diagonal_error,
        }
    )

    print(
        f"✓ SNPs      : {m:,}"
    )
    print(
        f"✓ Genotype  : "
        f"{mars_genotype.shape[0]:,} × "
        f"{mars_genotype.shape[1]:,}"
    )
    print(
        f"✓ Z-scores  : "
        f"{len(bin_z):,}"
    )
    print(
        f"✓ LD        : "
        f"{bin_ld.shape[0]:,} × "
        f"{bin_ld.shape[1]:,}"
    )
    print(
        f"✓ Output    : "
        f"{bin_dir}"
    )


# ============================================================
# 12. Save summary
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_file = (
    OUTPUT_DIR
    / "chr22_bin_mars_input_summary.tsv"
)

summary_df.to_csv(
    summary_file,
    sep="\t",
    index=False
)


# ============================================================
# 13. Final validation
# ============================================================

print("\n" + "=" * 60)
print("BIN-LEVEL MARS INPUT SUMMARY")
print("=" * 60)

print(
    summary_df[
        [
            "bin_id",
            "n_snps",
            "n_samples"
        ]
    ].to_string(
        index=False
    )
)

print("\n" + "=" * 60)
print("BIN-LEVEL MARS INPUT PREPARATION COMPLETED")
print("=" * 60)

print(
    f"Bins created : "
    f"{len(summary_df)}"
)

print(
    f"Output dir   : "
    f"{OUTPUT_DIR}"
)

print(
    f"Summary      : "
    f"{summary_file}"
)

print(
    "\n✓ All bin-level inputs prepared successfully"
)
