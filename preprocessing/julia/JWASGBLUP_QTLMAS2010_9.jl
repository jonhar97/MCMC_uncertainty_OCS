# =============================================================================
# GBLUP via JWAS (MCMC) — QTL-MAS 2010 Case Study — BIVARIATE MODEL
# =============================================================================
#
# Model (bivariate):
#   y1 = intercept + gender + genotypes + e    (continuous trait)
#   y2 = intercept + gender + genotypes + e    (binary trait, threshold model)
#
# Fixed:  intercept, gender (linear covariate 0/1)
# Random: genomic breeding value via GRM (VanRaden method 1)
#
# Key outputs for downstream robust OCS:
#   1. Posterior mean EBVs for both traits
#   2. Full MCMC EBV chains per trait — used to build Σ_r
#   3. Posterior covariance matrix Σ_r (index trait combining y1 and y2)
#   4. Accuracy: cor(EBV, TBV) for phenotyped / non-phenotyped, both traits
#
# INPUT FILES:
#   tbv.txt           — comma-delimited, no header
#                       col 1:  individual ID (integer, all 3226)
#                       col 2:  sex (0/1)
#                       col 3:  phenotype Q trait (observed)
#                       col 4:  phenotype B trait (observed binary)
#                       col 5:  TBV for Q trait (y1) — sex-dependent imprinting
#                       col 12: liability for B trait (TBV + residual)
#                       col 13: TBV for B trait (y2) — used for accuracy
#
#   phenotypes.txt    — comma-delimited, no header, PHENOTYPED ANIMALS ONLY
#                       col 1: individual ID
#                       col 2: phenotype y1 (continuous)
#                       col 3: phenotype y2 (binary 0/1)
#                       col 4: gender (0/1)
#
#   QTLMAS2010gen.txt — comma-delimited, no header, no ID column
#                       rows = all 3226 individuals (same order as TBV file)
#                       cols = marker dosage 0/1/2
#
# OUTPUT FILES (written to SAVE_DIR):
#   phenotypes_jwas.csv         — JWAS-ready phenotype file (all 3226, NA for unphenotyped)
#   GRM_QTLMAS_jwas.txt         — GRM in JWAS rowID format
#   EBV_y1_posterior_means.csv  — ID, posterior mean/SD EBV, TBV, phenotyped flag
#   EBV_y2_posterior_means.csv  — same for binary trait
#   EBV_y1_mcmc_chains.csv      — full MCMC chains for y1
#   EBV_y2_mcmc_chains.csv      — full MCMC chains for y2
#   Sigma_r_y1.csv              — posterior covariance matrix of EBVs, trait 1
#   Sigma_r_y2.csv              — posterior covariance matrix of EBVs, trait 2
#   PEV_diagonal.csv            — per-individual PEV for both traits
#   accuracy_report.txt         — cor(EBV, TBV) for both traits
#   variance_components.csv     — posterior h², genetic/residual variances
# =============================================================================

using JWAS, DataFrames, CSV, Statistics, LinearAlgebra, Printf, DelimitedFiles

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

BASE_DIR = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\QTLMAS\\"
SAVE_DIR = joinpath(BASE_DIR, "Save")
mkpath(SAVE_DIR)

# Input files
tbv_file   = joinpath(BASE_DIR, "tbv.txt")
pheno_file = joinpath(BASE_DIR, "phenotypes.txt")
geno_file  = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\QTLMAS\\QTLMAS2010gen\\QTLMAS2010gen.txt"

# Intermediate files (built by this script)
pheno_jwas_file = joinpath(SAVE_DIR, "phenotypes_jwas.csv")
grm_jwas_file   = joinpath(SAVE_DIR, "GRM_QTLMAS_jwas.txt")

# Output files
accuracy_out = joinpath(SAVE_DIR, "accuracy_report.txt")
varcomp_out  = joinpath(SAVE_DIR, "variance_components.csv")

# MCMC settings
CHAIN_LENGTH = 52_000   # increase to 50_000 for final runs
BURNIN       = 2_000    # iterations to discard (not saved)
THIN         = 10       # thinning: save every 10th post-burnin sample

println("=" ^ 70)
println("QTL-MAS 2010 — BIVARIATE GBLUP / JWAS (MCMC)")
println("=" ^ 70)

# =============================================================================
# 1. LOAD TBV FILE (master reference — all 3226 individuals)
# =============================================================================
# Format: ID, gender, TBV_y1, TBV_y2  (comma-delimited, no header)
println("\n[1] Loading TBV reference file (all individuals)...")

tbv_raw    = readdlm(tbv_file, ',', header=false)
all_ids    = Int.(tbv_raw[:, 1])
all_gender = Float64.(tbv_raw[:, 2])   # col 2: sex
all_tbv_y1 = Float64.(tbv_raw[:, 5])  # col 5: TBV Q trait (sex-dependent imprinting)
all_tbv_y2 = Float64.(tbv_raw[:, 13]) # col 13: TBV B trait (liability scale, excl. residual)
n_total    = length(all_ids)

id_to_row = Dict(all_ids[i] => i for i in 1:n_total)

println("  Total individuals: $n_total")

# =============================================================================
# 2. LOAD GENOTYPE MATRIX AND BUILD GRM (VanRaden method 1)
# =============================================================================
# Format: comma-delimited, no header, no ID column
# Rows are individuals in order matching TBV file; values are dosage 0/1/2
println("\n[2] Loading genotype matrix and building GRM (VanRaden 2008)...")
println("  (Loading $(n_total) × ~9000+ marker matrix — may take a moment...)")

geno_raw = readdlm(geno_file, ',', Float64, header=false)

@assert size(geno_raw, 1) == n_total "Genotype row count ($(size(geno_raw,1))) ≠ TBV count ($n_total)."

M         = geno_raw
n_markers = size(M, 2)
println("  Genotype matrix: $(size(M,1)) individuals × $n_markers markers")
println("  Assuming row order matches TBV file (individual $(all_ids[1])…$(all_ids[end]))")

# VanRaden method 1
p     = sum(M, dims=1) ./ (n_total * 2)
Z     = M .- 1 .- 2 .* (p .- 0.5)
denom = 2 * sum(p .* (1 .- p))
G_mat = (Z * Z') ./ denom

println("  GRM computed.")
@printf("  Diagonal mean    = %.4f  (expect ≈ 1.0)\n", mean(diag(G_mat)))
@printf("  Off-diagonal mean = %.4f\n", (sum(G_mat) - sum(diag(G_mat))) / (n_total^2 - n_total))

# Save GRM in JWAS rowID format: first column = ID, then n_total GRM values
println("  Saving GRM for JWAS...")
open(grm_jwas_file, "w") do io
    for i in 1:n_total
        print(io, all_ids[i])
        for j in 1:n_total
            @printf(io, ",%.8f", G_mat[i, j])
        end
        println(io)
    end
end
println("  GRM saved → $grm_jwas_file")

# =============================================================================
# 3. BUILD JWAS-READY PHENOTYPE FILE (BIVARIATE)
# =============================================================================
# phenotypes.txt: ID, y1, y2, gender  (no header, phenotyped animals only)
# JWAS needs all 3226 individuals; unphenotyped latest gen gets NA for both traits.
println("\n[3] Building JWAS bivariate phenotype file...")

pheno_raw  = readdlm(pheno_file, ',', header=false)
pheno_ids  = Int.(pheno_raw[:, 1])
pheno_y1   = pheno_raw[:, 2]          # continuous trait — keep as-is
pheno_y2   = pheno_raw[:, 3]          # binary trait 0/1
# Gender comes from tbv.txt col 2 (all_gender) — not present in phenotypes.txt

# Lookup: ID → (y1, y2)  — gender looked up from all_gender via id_to_row
pheno_lookup = Dict(pheno_ids[i] => (pheno_y1[i], pheno_y2[i])
                    for i in eachindex(pheno_ids))

n_phenotyped   = length(pheno_ids)
n_unphenotyped = n_total - n_phenotyped
println("  Phenotyped   : $n_phenotyped")
println("  Unphenotyped : $n_unphenotyped  (will receive NA for both traits)")

# Write CSV — JWAS requires header
open(pheno_jwas_file, "w") do io
    println(io, "ID,y1,y2,gender")
    for id in all_ids
        if haskey(pheno_lookup, id)
            y1_val, y2_val = pheno_lookup[id]
            g_val = all_gender[id_to_row[id]]
            @printf(io, "%d,%.6f,%d,%.0f\n", id, Float64(y1_val), Int(y2_val) + 1, g_val)
        else
            g_val = all_gender[id_to_row[id]]
            @printf(io, "%d,NA,NA,%.0f\n", id, g_val)
        end
    end
end
println("  JWAS phenotype file saved → $pheno_jwas_file")

# =============================================================================
# 4. LOAD DATA INTO JWAS
# =============================================================================
println("\n[4] Loading data into JWAS...")

phenotypes = CSV.read(pheno_jwas_file, DataFrame, delim=',', header=true,
                      missingstrings=["NA"])
phenotypes[!, :ID] = string.(phenotypes[!, :ID])

println("  Rows: $(nrow(phenotypes))")
println("  y1 observed: $(sum(.!ismissing.(phenotypes.y1)))  |  missing: $(sum(ismissing.(phenotypes.y1)))")
println("  y2 observed: $(sum(.!ismissing.(phenotypes.y2)))  |  missing: $(sum(ismissing.(phenotypes.y2)))")

genotypes = get_genotypes(grm_jwas_file, separator=',', method="GBLUP",
                          header=false, rowID=true)
println("  GRM loaded into JWAS")

# =============================================================================
# 5. BUILD BIVARIATE MODEL
# =============================================================================
# y1: continuous Gaussian trait
# y2: binary trait — JWAS uses a threshold (probit) model via categorical option
println("\n[5] Building bivariate JWAS model...")

model_equation = "y1 = intercept + gender + genotypes
                  y2 = intercept + gender + genotypes"


model = build_model(model_equation,categorical_trait=["y2"])

set_covariate(model, "gender")   # gender as linear fixed covariate in both traits

# Declare y2 as categorical (binary) — triggers threshold/probit model in JWAS
#set_binary(model, "y2")

println("  Model equations:")
println("    y1 = intercept + gender + genotypes  [Gaussian]")
println("    y2 = intercept + gender + genotypes  [Binary/threshold]")

# =============================================================================
# 6. RUN MCMC
# =============================================================================
println("\n[6] Running JWAS MCMC...")
@printf("  Chain length : %d\n", CHAIN_LENGTH)
@printf("  Burn-in      : %d\n", BURNIN)
@printf("  Thin         : %d  (saving every %dth sample)\n", THIN, THIN)
@printf("  Effective samples (post-burnin, thinned): %d\n", (CHAIN_LENGTH - BURNIN) ÷ THIN)
println("  Output folder: $SAVE_DIR")
println("  (This will take several minutes for a bivariate model...)")

out = runMCMC(model, phenotypes,
              chain_length             = CHAIN_LENGTH,
              burnin                   = BURNIN,
              output_samples_frequency = THIN,
              output_folder            = SAVE_DIR,
              printout_frequency       = 1000)

println("\n  MCMC complete.")

# =============================================================================
# 7. EXTRACT POSTERIOR SUMMARIES
# =============================================================================
println("\n[7] Extracting posterior summaries...")

h2_df = out["heritability"]
println("\n  Heritability:")
println(h2_df)

# JWAS bivariate EBV keys: "EBV_y1" and "EBV_y2"
ebv_df_y1 = out["EBV_y1"]
ebv_df_y2 = out["EBV_y2"]

println("\n  EBV y1 (first 5 rows):")
println(first(ebv_df_y1, 5))
println("\n  EBV y2 (first 5 rows):")
println(first(ebv_df_y2, 5))

# =============================================================================
# 8. LOAD FULL MCMC CHAINS (per trait)
# =============================================================================
# JWAS writes per-trait chain files named:
#   MCMC_samples_for_y1_genotypes.csv
#   MCMC_samples_for_y2_genotypes.csv
println("\n[8] Loading MCMC EBV chains...")

function load_chain(trait_name::String, save_dir::String)
    # Try the standard JWAS bivariate naming convention first
    candidates = [
        joinpath(save_dir, "MCMC_samples_for_$(trait_name)_genotypes.csv"),
        joinpath(save_dir, "MCMC_samples_for_genotypes_$(trait_name).csv"),
    ]
    # Also accept any file containing both "MCMC_samples" and trait name
    all_files = filter(f -> startswith(f, "MCMC_samples") && occursin(trait_name, f),
                       readdir(save_dir))
    append!(candidates, joinpath.(save_dir, all_files))

    for path in candidates
        if isfile(path)
            df = CSV.read(path, DataFrame, header=true)
            println("  $trait_name chain: $(nrow(df)) samples × $(ncol(df)) individuals  [$path]")
            return df
        end
    end
    @warn "No MCMC chain file found for $trait_name in $save_dir\n" *
          "Files present: $(join(readdir(save_dir), ", "))"
    return nothing
end

chains_y1 = load_chain("y1", SAVE_DIR)
chains_y2 = load_chain("y2", SAVE_DIR)

# =============================================================================
# 9. ACCURACY: cor(EBV, TBV) for both traits
# =============================================================================
println("\n[9] Calculating accuracy (EBV vs TBV)...")

function build_accuracy(ebv_df, all_ids, all_tbv, pheno_lookup, id_to_row, trait_label)
    # JWAS bivariate EBV table columns: ID, EBV, PEV
    ebv_ids_str  = string.(ebv_df.ID)
    ebv_means_v  = Float64.(ebv_df.EBV)
    ebv_pev_v    = Float64.(ebv_df.PEV)          # PEV (not SD) — convert below
    ebv_by_id    = Dict(parse(Int, ebv_ids_str[i]) => ebv_means_v[i]
                        for i in eachindex(ebv_ids_str))
    ebv_sd_by_id = Dict(parse(Int, ebv_ids_str[i]) => sqrt(max(ebv_pev_v[i], 0.0))
                        for i in eachindex(ebv_ids_str))   # SD = sqrt(PEV)

    ids   = Int[];    ebv  = Float64[];  tbv   = Float64[]
    pheno = Bool[];   sds  = Float64[]

    for i in eachindex(all_ids)
        id = all_ids[i]
        haskey(ebv_by_id, id) || continue
        push!(ids,   id)
        push!(ebv,   ebv_by_id[id])
        push!(tbv,   all_tbv[i])
        push!(pheno, haskey(pheno_lookup, id))
        push!(sds,   get(ebv_sd_by_id, id, NaN))
    end

    r_phen = cor(ebv[pheno],   tbv[pheno])
    r_unph = cor(ebv[.!pheno], tbv[.!pheno])
    @printf("  %-4s  Phenotyped    : r(EBV,TBV) = %.4f  (n=%d)\n",
            trait_label, r_phen, sum(pheno))
    @printf("  %-4s  Non-phenotyped: r(EBV,TBV) = %.4f  (n=%d)\n",
            trait_label, r_unph, sum(.!pheno))

    return ids, ebv, tbv, pheno, sds, r_phen, r_unph
end

ids_y1, ebv_y1, tbv_y1, pheno_y1, sd_y1, r_phen_y1, r_unph_y1 =
    build_accuracy(ebv_df_y1, all_ids, all_tbv_y1, pheno_lookup, id_to_row, "y1")

ids_y2, ebv_y2, tbv_y2, pheno_y2, sd_y2, r_phen_y2, r_unph_y2 =
    build_accuracy(ebv_df_y2, all_ids, all_tbv_y2, pheno_lookup, id_to_row, "y2")

# =============================================================================
# 10. SAVE OUTPUTS
# =============================================================================
println("\n[10] Writing output files...")

# EBV posterior means — one file per trait
for (ids, ebv, tbv, pheno, sds, fname) in [
        (ids_y1, ebv_y1, tbv_y1, pheno_y1, sd_y1, joinpath(SAVE_DIR, "EBV_y1_posterior_means.csv")),
        (ids_y2, ebv_y2, tbv_y2, pheno_y2, sd_y2, joinpath(SAVE_DIR, "EBV_y2_posterior_means.csv"))]
    df = DataFrame(ID=ids, EBV_post_mean=ebv, EBV_post_SD=sds,   # SD = sqrt(PEV)
                   TBV=tbv, Phenotyped=Int.(pheno))
    CSV.write(fname, df)
    println("  EBV means → $fname")
end

# MCMC chains
for (chains, fname) in [
        (chains_y1, joinpath(SAVE_DIR, "EBV_y1_mcmc_chains.csv")),
        (chains_y2, joinpath(SAVE_DIR, "EBV_y2_mcmc_chains.csv"))]
    if !isnothing(chains)
        CSV.write(fname, chains)
        println("  MCMC chains → $fname")
    end
end

# Accuracy report
open(accuracy_out, "w") do io
    @printf(io, "Trait\tGroup\tn\tPearson_r_EBV_vs_TBV\n")
    @printf(io, "y1\tPhenotyped\t%d\t%.6f\n",     sum(pheno_y1),   r_phen_y1)
    @printf(io, "y1\tNon-phenotyped\t%d\t%.6f\n", sum(.!pheno_y1), r_unph_y1)
    @printf(io, "y2\tPhenotyped\t%d\t%.6f\n",     sum(pheno_y2),   r_phen_y2)
    @printf(io, "y2\tNon-phenotyped\t%d\t%.6f\n", sum(.!pheno_y2), r_unph_y2)
end
println("  Accuracy report → $accuracy_out")

# Variance components / heritability
CSV.write(varcomp_out, h2_df)
println("  Variance components → $varcomp_out")

# =============================================================================
# 11. POSTERIOR COVARIANCE MATRICES Σ_r (per trait, for robust OCS)
# =============================================================================
# For each trait, Σ_r[i,j] = posterior covariance of EBV_i and EBV_j.
# Used in the robust OCS objective: Maximize c'μ_r − λ·c'G·c − γ·c'Σ_r·c

function compute_and_save_sigma_r(chains::Union{DataFrame,Nothing},
                                  save_path::String, pev_label::String)
    isnothing(chains) && return nothing

    chain_mat     = Matrix{Float64}(chains)
    n_samp, n_ind = size(chain_mat)
    col_means     = mean(chain_mat, dims=1)
    centred       = chain_mat .- col_means
    Sigma_r       = (centred' * centred) ./ (n_samp - 1)

    @printf("  %s  diagonal mean PEV = %.6f  |  max PEV = %.6f\n",
            pev_label, mean(diag(Sigma_r)), maximum(diag(Sigma_r)))

    col_ids = names(chains)
    open(save_path, "w") do io
        println(io, "ID," * join(col_ids, ","))
        for i in 1:n_ind
            print(io, col_ids[i])
            for j in 1:n_ind
                @printf(io, ",%.8f", Sigma_r[i, j])
            end
            println(io)
        end
    end
    println("  Σ_r saved → $save_path")
    return Sigma_r, col_ids
end

println("\n[11] Computing posterior covariance matrices Σ_r...")

result_y1 = compute_and_save_sigma_r(chains_y1,
                joinpath(SAVE_DIR, "Sigma_r_y1.csv"), "y1")
result_y2 = compute_and_save_sigma_r(chains_y2,
                joinpath(SAVE_DIR, "Sigma_r_y2.csv"), "y2")

# PEV diagonal summary — both traits in one file
if !isnothing(result_y1) && !isnothing(result_y2)
    Sigma_r_y1, col_ids_y1 = result_y1
    Sigma_r_y2, col_ids_y2 = result_y2
    pev_df = DataFrame(
        ID           = col_ids_y1,
        PEV_y1       = diag(Sigma_r_y1),
        postSD_y1    = sqrt.(diag(Sigma_r_y1)),
        PEV_y2       = diag(Sigma_r_y2),
        postSD_y2    = sqrt.(diag(Sigma_r_y2))
    )
    CSV.write(joinpath(SAVE_DIR, "PEV_diagonal.csv"), pev_df)
    println("  PEV diagonal → $(joinpath(SAVE_DIR, "PEV_diagonal.csv"))")
end

# =============================================================================
# DONE
# =============================================================================
println("\n" * "=" ^ 70)
println("ANALYSIS COMPLETE")
println("=" ^ 70)
println("""
Key outputs for robust OCS:
  EBV posterior means  : EBV_y1_posterior_means.csv / EBV_y2_posterior_means.csv
  MCMC chains          : EBV_y1_mcmc_chains.csv     / EBV_y2_mcmc_chains.csv
  Σ_r matrices         : Sigma_r_y1.csv             / Sigma_r_y2.csv
  PEV summary          : PEV_diagonal.csv
  Accuracy             : $accuracy_out

Robust OCS objective (index trait combining y1 and y2):
  Maximize: c'μ_r − λ·c'G·c − γ·c'Σ_r·c
  subject to: sum(c) = 1,  c ≥ 0,  0.5·c'G·c ≤ Θ

Comparison targets:
  MAP-OCS on EBV   : μ_r,          γ = 0  (naive)
  Robust-OCS       : μ_r + Σ_r,    γ > 0  (sweep γ for frontier)
  MAP-OCS on TBV   : all_tbv_y1/y2, γ = 0  (gold standard)
""")
