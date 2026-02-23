"""
Simulate Example Data for MCMC-OCS Uncertainty Analysis
========================================================

Generates realistic synthetic data resembling the Norway spruce dataset:
  - n = 200 individuals across 20 full-sib families (10 per family)
  - Genomic relationship matrix G (mimicking SNP-derived GRM)
  - Posterior MCMC EBV samples for a single index trait
  - True breeding values with heritability h² ≈ 0.35

Run this script once to create data files used by run_example.jl.

Author: Jon Ahlinder (Skogforsk) / Patrik Waldmann
"""

using Random, LinearAlgebra, Statistics, CSV, DataFrames, DelimitedFiles

Random.seed!(2026)

# ============================================================================
# PARAMETERS (mirroring Norway spruce setup)
# ============================================================================
N_IND       = 200        # individuals
N_FAM       = 20         # full-sib families
FAM_SIZE    = 10         # individuals per family
N_MCMC      = 500        # MCMC posterior draws
H2          = 0.35       # heritability
THETA       = 0.02       # coancestry constraint for OCS
SIGMA2_G    = 1.0        # genetic variance
SIGMA2_E    = SIGMA2_G * (1 - H2) / H2   # residual variance

SAVE_PATH   = @__DIR__   # save to the example/ directory

println("="^70)
println("SIMULATING EXAMPLE DATA")
println("="^70)
println("  Individuals : $N_IND")
println("  Families    : $N_FAM ($(FAM_SIZE) per family)")
println("  MCMC draws  : $N_MCMC")
println("  h²          : $H2")
println("  θ (OCS)     : $THETA")
println()

# ============================================================================
# 1. FAMILY AND INDIVIDUAL STRUCTURE
# ============================================================================
println("Step 1: Building family structure...")

family_ids = repeat(1:N_FAM, inner=FAM_SIZE)   # family label per individual
ind_ids    = 1:N_IND

# Save phenotype metadata
meta_df = DataFrame(
    ID       = collect(ind_ids),
    Family   = family_ids
)
CSV.write(joinpath(SAVE_PATH, "example_phenotypes.csv"), meta_df)
println("  ✔ Saved example_phenotypes.csv")

# ============================================================================
# 2. GENOMIC RELATIONSHIP MATRIX (G)
# ============================================================================
println("\nStep 2: Constructing G matrix...")

# Simulate a block-structured GRM:
#   - High within-family relatedness (~0.5 for full sibs)
#   - Low between-family relatedness (~0.02 population mean)

G = fill(0.02, N_IND, N_IND)   # base off-diagonal

for fam in 1:N_FAM
    idx = findall(family_ids .== fam)
    for i in idx, j in idx
        if i == j
            G[i, j] = 1.0 + 0.05 * randn()   # diagonal ≈ 1
        else
            G[i, j] = 0.5 + 0.05 * randn()   # full-sib ≈ 0.5
        end
    end
end

# Ensure positive definite
G = G + 0.01 * I(N_IND)
# Symmetrise
G = (G + G') / 2

# Write G matrix with ID column (JWAS format)
G_with_ids = hcat(collect(Float64, ind_ids), G)
open(joinpath(SAVE_PATH, "example_G_matrix.txt"), "w") do io
    writedlm(io, G_with_ids, ',')
end
println("  ✔ Saved example_G_matrix.txt  ($N_IND × $N_IND)")

# ============================================================================
# 3. TRUE BREEDING VALUES AND PHENOTYPES
# ============================================================================
println("\nStep 3: Simulating true breeding values...")

# Family genetic effects
fam_effects = randn(N_FAM) .* sqrt(SIGMA2_G / 2)

# Within-family Mendelian sampling
true_bv = [fam_effects[family_ids[i]] + randn() * sqrt(SIGMA2_G / 2)
           for i in 1:N_IND]

# Phenotypic observations
phenotypes = true_bv .+ randn(N_IND) .* sqrt(SIGMA2_E)

# Save
pheno_df = DataFrame(
    ID       = collect(ind_ids),
    Family   = family_ids,
    Phenotype = round.(phenotypes, digits=4),
    TrueBV   = round.(true_bv, digits=4)
)
CSV.write(joinpath(SAVE_PATH, "example_phenotypes.csv"), pheno_df)
println("  ✔ Saved example_phenotypes.csv")
println("    True BV range: [$(round(minimum(true_bv),digits=2)), $(round(maximum(true_bv),digits=2))]")

# ============================================================================
# 4. POSTERIOR MCMC EBV SAMPLES (simulated JWAS output)
# ============================================================================
println("\nStep 4: Simulating posterior MCMC EBV samples...")

# In JWAS output: rows = individuals, columns = MCMC iterations
# Posterior mean ≈ shrunk estimate of true BV
# Posterior SD reflects uncertainty (larger for small families)

# Shrinkage factor (mimics BLUP shrinkage)
shrinkage = H2

# Family size affects posterior SD: smaller family = more uncertainty
fam_counts = [sum(family_ids .== f) for f in 1:N_FAM]
posterior_sd = [0.3 + 0.4 / sqrt(fam_counts[family_ids[i]]) for i in 1:N_IND]

# Generate MCMC samples: individual × iteration matrix
posterior_mean = shrinkage .* true_bv .+ (1 - shrinkage) .* mean(true_bv)

mcmc_samples = zeros(N_IND, N_MCMC)
for iter in 1:N_MCMC
    # Each iteration: posterior mean + individual noise + global noise
    mcmc_samples[:, iter] = posterior_mean .+
        randn(N_IND) .* posterior_sd .+
        randn() * 0.05   # small chain noise
end

# Save as CSV (JWAS MCMC_samples_EBV format)
open(joinpath(SAVE_PATH, "example_MCMC_EBV_samples.txt"), "w") do io
    writedlm(io, mcmc_samples, ',')
end
println("  ✔ Saved example_MCMC_EBV_samples.txt  ($N_IND rows × $N_MCMC columns)")
println("    Posterior SD range: [$(round(minimum(posterior_sd),digits=3)), $(round(maximum(posterior_sd),digits=3))]")

# ============================================================================
# 5. MAP EBV (posterior means — used for MAP-OCS)
# ============================================================================
println("\nStep 5: Calculating MAP EBVs (posterior means)...")

map_ebv = vec(mean(mcmc_samples, dims=2))

map_df = DataFrame(
    ID     = collect(ind_ids),
    Family = family_ids,
    MAP_EBV = round.(map_ebv, digits=6)
)
CSV.write(joinpath(SAVE_PATH, "example_MAP_EBV.csv"), map_df)
println("  ✔ Saved example_MAP_EBV.csv")
println("    MAP EBV range: [$(round(minimum(map_ebv),digits=2)), $(round(maximum(map_ebv),digits=2))]")

# ============================================================================
# SUMMARY
# ============================================================================
println()
println("="^70)
println("SIMULATION COMPLETE")
println("="^70)
println()
println("Generated files:")
println("  example_phenotypes.csv         — individual metadata + phenotypes")
println("  example_G_matrix.txt           — genomic relationship matrix (JWAS format)")
println("  example_MCMC_EBV_samples.txt   — posterior EBV samples (individuals × iterations)")
println("  example_MAP_EBV.csv            — MAP EBV (posterior means)")
println()
println("Next step: run  include(\"run_example.jl\")")
println()

# Return key objects for interactive use
(; G, map_ebv, mcmc_samples, family_ids, posterior_sd, N_IND, N_MCMC, THETA)
