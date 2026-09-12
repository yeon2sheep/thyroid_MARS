from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

GENO_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "mars_input"
    / "chr22_EAS504_genotype_final.tsv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "mars_input"
)

LD_RAW_FILE = (
    OUTPUT_DIR
    / "chr22_LD_raw.tsv"
)

LD_FINAL_FILE = (
    OUTPUT_DIR
    / "chr22_LD_final.tsv"
)


# ============================================================
# 2. Load genotype
# ============================================================

print("=" * 60)
print("Loading final genotype matrix")
print("=" * 60)

geno = pd.read_csv(
    GENO_FILE,
    sep="\t"
)

assert "variant_key" in geno.columns

variant_keys = geno["variant_key"].tolist()

sample_columns = geno.columns[1:]

x = geno[sample_columns].to_numpy(
    dtype=np.float64
)

n_snps, n_samples = x.shape

print(f"SNPs    : {n_snps:,}")
print(f"Samples : {n_samples:,}")

assert n_snps == 1799
assert n_samples == 504

print("✓ Genotype matrix dimensions are correct")


# ============================================================
# 3. Validate genotype
# ============================================================

if np.isnan(x).any():
    raise ValueError(
        "Genotype matrix contains NaN values."
    )

if np.isinf(x).any():
    raise ValueError(
        "Genotype matrix contains Inf values."
    )

if x.min() < 0 or x.max() > 2:
    raise ValueError(
        "Genotype values are outside 0~2."
    )

print("✓ No NaN/Inf")
print("✓ Genotype values within 0~2")


# ============================================================
# 4. Calculate LD correlation matrix
# ============================================================

print("\n" + "=" * 60)
print("Calculating LD correlation matrix")
print("=" * 60)

print("Using Pearson correlation between SNP genotype vectors...")
print("Equivalent to: cor(t(x))")

# Each row = one SNP
# Each column = one EAS sample
#
# np.corrcoef(x) calculates correlation between rows
# and therefore corresponds to R:
#
# cor(t(x))

ld = np.corrcoef(x)

print(
    f"Raw LD matrix shape : "
    f"{ld.shape[0]:,} × {ld.shape[1]:,}"
)


# ============================================================
# 5. Validate raw LD matrix
# ============================================================

if np.isnan(ld).any():
    raise ValueError(
        "Raw LD matrix contains NaN values."
    )

if np.isinf(ld).any():
    raise ValueError(
        "Raw LD matrix contains Inf values."
    )

print("✓ Raw LD contains no NaN/Inf")

# Symmetry check
symmetry_error = np.max(
    np.abs(ld - ld.T)
)

print(
    f"Maximum symmetry error : "
    f"{symmetry_error:.3e}"
)

assert symmetry_error < 1e-10

print("✓ LD matrix is symmetric")


# ============================================================
# 6. Diagonal check
# ============================================================

diagonal = np.diag(ld)

max_diagonal_error = np.max(
    np.abs(diagonal - 1.0)
)

print(
    f"Maximum diagonal error : "
    f"{max_diagonal_error:.3e}"
)

assert max_diagonal_error < 1e-10

print("✓ LD diagonal = 1")


# ============================================================
# 7. Eigenvalue check
# ============================================================

print("\nChecking positive definiteness...")

eigenvalues = np.linalg.eigvalsh(ld)

min_eigenvalue = eigenvalues.min()
max_eigenvalue = eigenvalues.max()

print(
    f"Minimum eigenvalue : "
    f"{min_eigenvalue:.6e}"
)

print(
    f"Maximum eigenvalue : "
    f"{max_eigenvalue:.6e}"
)


# ============================================================
# 8. Save raw LD
# ============================================================

print("\nSaving raw LD matrix...")

ld_df = pd.DataFrame(
    ld
)

ld_df.to_csv(
    LD_RAW_FILE,
    sep="\t",
    header=False,
    index=False
)

print(LD_RAW_FILE)


# ============================================================
# 9. Near positive-definite correction
# ============================================================

print("\n" + "=" * 60)
print("Near positive-definite correction")
print("=" * 60)

if min_eigenvalue >= 0:
    print(
        "Raw LD matrix is already positive semidefinite."
    )

    ld_final = ld.copy()

else:
    print(
        "Raw LD matrix has negative eigenvalues."
    )

    print(
        "A nearPD-style correction is required."
    )

    # --------------------------------------------------------
    # Eigenvalue clipping
    # --------------------------------------------------------
    #
    # MARS generateLD.R uses:
    #
    # nearPD(ld)$mat
    #
    # Here we use an eigenvalue-based nearest
    # positive-semidefinite approximation.
    #
    # Small negative eigenvalues are clipped to zero.
    #

    eigenvalues, eigenvectors = np.linalg.eigh(ld)

    eigenvalues_clipped = np.maximum(
        eigenvalues,
        0
    )

    ld_final = (
        eigenvectors
        @ np.diag(eigenvalues_clipped)
        @ eigenvectors.T
    )

    # Numerical symmetrization
    ld_final = (
        ld_final + ld_final.T
    ) / 2

    # Normalize diagonal back to 1
    d = np.sqrt(
        np.diag(ld_final)
    )

    ld_final = (
        ld_final
        / np.outer(d, d)
    )

    print("✓ Eigenvalue correction completed")


# ============================================================
# 10. Final LD validation
# ============================================================

print("\n" + "=" * 60)
print("Final LD validation")
print("=" * 60)

final_symmetry_error = np.max(
    np.abs(ld_final - ld_final.T)
)

print(
    f"Symmetry error : "
    f"{final_symmetry_error:.3e}"
)

assert final_symmetry_error < 1e-10

final_diagonal = np.diag(ld_final)

final_diagonal_error = np.max(
    np.abs(final_diagonal - 1.0)
)

print(
    f"Diagonal error : "
    f"{final_diagonal_error:.3e}"
)

assert final_diagonal_error < 1e-10

final_eigenvalues = np.linalg.eigvalsh(
    ld_final
)

final_min_eigenvalue = (
    final_eigenvalues.min()
)

print(
    f"Minimum eigenvalue : "
    f"{final_min_eigenvalue:.6e}"
)

assert final_min_eigenvalue >= -1e-8

print("✓ Final LD matrix is positive semidefinite")
print("✓ Final LD matrix is symmetric")
print("✓ Final LD diagonal = 1")


# ============================================================
# 11. Check SNP order
# ============================================================

assert ld_final.shape == (
    n_snps,
    n_snps
)

print(
    f"\n✓ LD dimensions = "
    f"{n_snps:,} × {n_snps:,}"
)

print(
    "✓ LD SNP order corresponds to genotype file order"
)


# ============================================================
# 12. Save final LD matrix
# ============================================================

print("\nSaving final LD matrix...")

pd.DataFrame(ld_final).to_csv(
    LD_FINAL_FILE,
    sep="\t",
    header=False,
    index=False
)

print(LD_FINAL_FILE)


# ============================================================
# 13. Final summary
# ============================================================

print("\n" + "=" * 60)
print("LD GENERATION COMPLETED")
print("=" * 60)

print(f"SNPs             : {n_snps:,}")
print(f"EAS samples      : {n_samples:,}")
print(
    f"LD matrix        : "
    f"{n_snps:,} × {n_snps:,}"
)

print(
    f"Raw LD           : "
    f"{LD_RAW_FILE}"
)

print(
    f"Final LD         : "
    f"{LD_FINAL_FILE}"
)

print("\n✓ LD generation completed successfully")