from pathlib import Path
import re
import pandas as pd


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

# Reference annotation
GTF_FILE = PROJECT_DIR / "data" / "reference" / "genes.gtf"

# Final data from previous pipeline
FINAL_DIR = PROJECT_DIR / "data" / "processed" / "final"

SNP_FILE = FINAL_DIR / "chr22_final_snps.tsv"
Z_FILE = FINAL_DIR / "chr22_final_z.tsv"
GENOTYPE_FILE = FINAL_DIR / "chr22_final_genotype.tsv"

# Gene-bin definition
GENE_BIN_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "gene_bins"
    / "chr22_gene_bins_pm50.tsv"
)

# Output
OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "gene_bin_inputs"
)

MEMBERSHIP_FILE = OUTPUT_DIR / "chr22_gene_bin_membership.tsv"
SUMMARY_FILE = OUTPUT_DIR / "chr22_gene_bin_summary.tsv"

CHROMOSOME = "22"
FLANK_SIZE = 50


# ============================================================
# 2. Helper: parse GTF attributes
# ============================================================

def get_attribute(attributes, key):
    """
    Extract one attribute from the GTF attribute column.

    Example:
        gene_id "ENSG00000294541";
        gene_name "ABC1";

    Returns:
        ABC1
    """

    match = re.search(rf'{key}\s+"([^"]+)"', attributes)

    if match:
        return match.group(1)

    return None


# ============================================================
# 3. Create gene bins from GTF
# ============================================================

def make_gene_bins():

    print("=" * 60)
    print("STEP 1. Creating gene bins")
    print("=" * 60)

    gene_bins = []

    with open(GTF_FILE, "r") as f:

        for line in f:

            if line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")

            if len(fields) != 9:
                continue

            chrom = fields[0]
            feature = fields[2]

            # Keep chromosome 22
            if chrom != CHROMOSOME:
                continue

            # Keep gene features only
            if feature != "gene":
                continue

            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            attributes = fields[8]

            gene_id = get_attribute(attributes, "gene_id")
            gene_name = get_attribute(attributes, "gene_name")
            gene_biotype = get_attribute(
                attributes,
                "gene_biotype"
            )

            # gene +/- 50 bp
            bin_start = max(1, start - FLANK_SIZE)
            bin_end = end + FLANK_SIZE

            gene_bins.append(
                {
                    "gene_id": gene_id,
                    "gene_name": gene_name,
                    "gene_biotype": gene_biotype,
                    "chrom": chrom,
                    "gene_start": start,
                    "gene_end": end,
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    "strand": strand,
                }
            )

    gene_bins = pd.DataFrame(gene_bins)

    gene_bins = gene_bins.sort_values(
        by=["chrom", "bin_start", "bin_end", "gene_id"]
    ).reset_index(drop=True)

    gene_bins.insert(
        0,
        "bin_id",
        [
            f"GENE_BIN_{i:05d}"
            for i in range(1, len(gene_bins) + 1)
        ]
    )

    print(f"Total chr22 genes: {len(gene_bins):,}")

    return gene_bins


# ============================================================
# 4. Load final SNP information
# ============================================================

def load_final_snps():

    print()
    print("=" * 60)
    print("STEP 2. Loading final SNP information")
    print("=" * 60)

    snps = pd.read_csv(
        SNP_FILE,
        sep="\t"
    )

    # Keep only the columns needed for bin matching
    snps = snps[
        [
            "variant_key",
            "chromosome",
            "base_pair_location",
        ]
    ].copy()

    snps["base_pair_location"] = pd.to_numeric(
        snps["base_pair_location"]
    )

    print(f"Final SNPs: {len(snps):,}")

    return snps


# ============================================================
# 5. Load Z scores
# ============================================================

def load_z():

    print()
    print("=" * 60)
    print("STEP 3. Loading Z scores")
    print("=" * 60)

    z = pd.read_csv(
        Z_FILE,
        sep="\t"
    )

    z = z[
        [
            "variant_key",
            "z",
        ]
    ].copy()

    print(f"Z-score records: {len(z):,}")

    return z


# ============================================================
# 6. Load genotype
# ============================================================

def load_genotype():

    print()
    print("=" * 60)
    print("STEP 4. Loading genotype")
    print("=" * 60)

    genotype = pd.read_csv(
        GENOTYPE_FILE,
        sep="\t"
    )

    print(
        f"Genotype variants: {len(genotype):,}"
    )

    print(
        f"Genotype samples: "
        f"{genotype.shape[1] - 1:,}"
    )

    return genotype


# ============================================================
# 7. Match SNPs to gene bins
# ============================================================

def match_snps_to_bins(gene_bins, snps):

    print()
    print("=" * 60)
    print("STEP 5. Matching SNPs to gene bins")
    print("=" * 60)

    membership = []

    snp_positions = snps[
        [
            "variant_key",
            "base_pair_location",
        ]
    ]

    for _, gene in gene_bins.iterrows():

        start = gene["bin_start"]
        end = gene["bin_end"]

        matched = snp_positions[
            (snp_positions["base_pair_location"] >= start)
            &
            (snp_positions["base_pair_location"] <= end)
        ]

        for _, snp in matched.iterrows():

            membership.append(
                {
                    "bin_id": gene["bin_id"],
                    "gene_id": gene["gene_id"],
                    "gene_name": gene["gene_name"],
                    "bin_start": start,
                    "bin_end": end,
                    "variant_key": snp["variant_key"],
                    "base_pair_location": snp[
                        "base_pair_location"
                    ],
                }
            )

    membership = pd.DataFrame(membership)

    print(
        f"Total gene-SNP assignments: "
        f"{len(membership):,}"
    )

    print(
        f"Gene bins containing >=1 SNP: "
        f"{membership['bin_id'].nunique():,}"
    )

    return membership


# ============================================================
# 8. Add Z scores
# ============================================================

def add_z_scores(membership, z):

    print()
    print("=" * 60)
    print("STEP 6. Adding Z scores")
    print("=" * 60)

    membership = membership.merge(
        z,
        on="variant_key",
        how="left",
        validate="many_to_one"
    )

    missing_z = membership["z"].isna().sum()

    print(f"Missing Z scores: {missing_z:,}")

    return membership


# ============================================================
# 9. Create summary
# ============================================================

def make_summary(gene_bins, membership):

    print()
    print("=" * 60)
    print("STEP 7. Creating gene-bin summary")
    print("=" * 60)

    # Number of SNPs per gene bin
    snp_counts = (
        membership
        .groupby("bin_id")
        .size()
        .rename("n_snps")
    )

    summary = gene_bins.merge(
        snp_counts,
        on="bin_id",
        how="left"
    )

    summary["n_snps"] = (
        summary["n_snps"]
        .fillna(0)
        .astype(int)
    )

    summary["has_snps"] = (
        summary["n_snps"] > 0
    )

    return summary


# ============================================================
# 10. Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("Starting gene-based bin pipeline")
    print(f"Gene definition: +/- {FLANK_SIZE} bp")
    print()

    # 1. Gene bins
    gene_bins = make_gene_bins()

    # Save gene bins
    GENE_BIN_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    gene_bins.to_csv(
        GENE_BIN_FILE,
        sep="\t",
        index=False
    )

    print(
        f"Gene bins saved: {GENE_BIN_FILE}"
    )

    # 2. Final SNPs
    snps = load_final_snps()

    # 3. Z scores
    z = load_z()

    # 4. Genotype
    genotype = load_genotype()

    # Make sure genotype contains the same variants
    genotype_variant_keys = set(
        genotype["variant_key"]
    )

    # 5. Match SNPs to gene bins
    membership = match_snps_to_bins(
        gene_bins,
        snps
    )

    # 6. Add Z scores
    membership = add_z_scores(
        membership,
        z
    )

    # Check whether genotype exists for matched SNPs
    membership["has_genotype"] = (
        membership["variant_key"]
        .isin(genotype_variant_keys)
    )

    missing_genotype = (
        ~membership["has_genotype"]
    ).sum()

    print(
        f"Missing genotype: "
        f"{missing_genotype:,}"
    )

    # Save membership table
    membership.to_csv(
        MEMBERSHIP_FILE,
        sep="\t",
        index=False
    )

    # 7. Summary
    summary = make_summary(
        gene_bins,
        membership
    )

    summary.to_csv(
        SUMMARY_FILE,
        sep="\t",
        index=False
    )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    print(
        f"Total gene bins: "
        f"{len(gene_bins):,}"
    )

    print(
        f"Gene-SNP assignments: "
        f"{len(membership):,}"
    )

    print(
        f"Gene bins with SNPs: "
        f"{summary['has_snps'].sum():,}"
    )

    print(
        f"Gene bins without SNPs: "
        f"{(~summary['has_snps']).sum():,}"
    )

    print()
    print("Output files:")
    print(f"1. {GENE_BIN_FILE}")
    print(f"2. {MEMBERSHIP_FILE}")
    print(f"3. {SUMMARY_FILE}")


if __name__ == "__main__":
    main()