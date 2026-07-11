# =============================================================================
# CVaR-OCS — Stochastic Optimum Contribution Selection
# QTL-MAS 2010 Case Study
# =============================================================================
#
# Implements the CVaR-aware OCS formulation from Ahlinder & Waldmann:
#
#   max_{c, η, z}   (1/l) Σ_j c'g^(j)  +  μ [ η − 1/((1−α)l) Σ_j z_j ]
#   subject to:
#     z_j ≥ η − c'g^(j),   j = 1,…,l      (shortfall constraints)
#     z_j ≥ 0,              j = 1,…,l
#     (1/2) c'Σ c ≤ Θ                       (coancestry / inbreeding bound)
#     c'1 = 1                               (contributions sum to 1)
#     c ≥ 0
#
# Decision variables:
#   c  ∈ R^n   — individual contributions  (n = all candidates)
#   η  ∈ R     — VaR threshold (optimised jointly)
#   z  ∈ R^l   — per-scenario shortfall below η
#
# Key parameters:
#   α   ∈ (0,1)  — CVaR confidence level (e.g. 0.95 → protects worst 5%)
#   μ   ≥ 0      — risk-aversion weight  (μ=0 → posterior-mean MAP-OCS)
#   Θ            — maximum group coancestry (hard constraint)
#   l            — number of MCMC scenarios (posterior draws)
#
# Inputs (from JWASGBLUP_QTLMAS2010_9.jl):
#   EBV_y1_posterior_means.csv   — posterior mean EBVs, trait 1
#   EBV_y2_posterior_means.csv   — posterior mean EBVs, trait 2
#   EBV_y1_mcmc_chains.csv       — MCMC chain, trait 1  (rows=iterations, cols=individuals)
#   EBV_y2_mcmc_chains.csv       — MCMC chain, trait 2
#   GRM_QTLMAS_jwas.txt          — genomic relationship matrix (VanRaden)
#
# Outputs (written to SAVE_DIR):
#   cvar_ocs_solutions.csv       — contributions for MAP-OCS and each (μ, α) combination
#   cvar_ocs_frontier.csv        — expected gain vs CVaR-gain for all settings
#   cvar_ocs_summary.txt         — human-readable summary
#
# Author: Jon Ahlinder (Skogforsk) / Ahlinder & Waldmann
# =============================================================================

using CSV, DataFrames, Statistics, LinearAlgebra
using JuMP, COSMO
using DelimitedFiles: readdlm
using Printf

# =============================================================================
# 0. CONFIGURATION — edit these paths and parameters
# =============================================================================

BASE_DIR = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\QTLMAS\\"
SAVE_DIR = joinpath(BASE_DIR, "Save6")
SAVE_DIR2 = joinpath(BASE_DIR, "Save")

# Input files produced by JWASGBLUP_QTLMAS2010_9.jl
EBV_Y1_MEANS  = joinpath(SAVE_DIR, "EBV_y1.txt")
EBV_Y2_MEANS  = joinpath(SAVE_DIR, "EBV_y2.txt")
EBV_Y1_CHAINS = joinpath(SAVE_DIR, "MCMC_samples_EBV_y1.txt")
EBV_Y2_CHAINS = joinpath(SAVE_DIR, "MCMC_samples_EBV_y2.txt")
GRM_FILE      = joinpath(SAVE_DIR2, "GRM_QTLMAS_jwas.txt")

# Source files to identify last-generation animals (all IDs minus phenotyped IDs)
TBV_FILE   = joinpath(BASE_DIR, "tbv.txt")        # col 1 = all 3226 IDs, comma-delim, no header
PHENO_FILE = joinpath(BASE_DIR, "phenotypes.txt") # col 1 = phenotyped IDs only, comma-delim, no header

# OCS constraint
THETA = 0.03            # coancestry upper bound (Θ)
theta_folder = "Theta$(Int(round(THETA*100)))"  # e.g. Theta3 for 0.03 — built dynamically so output path always matches THETA

# CVaR confidence level(s) to evaluate
ALPHA_VALUES = [0.90, 0.95, 0.99]

# Risk-aversion weights to sweep (μ = 0 recovers posterior-mean MAP-OCS)
MU_VALUES = [0.0, 0.5, 0.75,1.0, 1.25, 1.5, 1.75, 2.0, 5.0, 10.0]

# COSMO solver tolerance / iterations
COSMO_MAX_ITER = 50_000
COSMO_EPS_ABS  = 1e-5
COSMO_EPS_REL  = 1e-5

# Contribution threshold for "selected"
CONTRIBUTION_THRESHOLD = 1e-4

println("=" ^ 70)
println("CVaR-OCS — Stochastic Optimum Contribution Selection")
println("QTL-MAS 2010")
println("=" ^ 70)

# =============================================================================
# 1. LOAD GRM
# =============================================================================
println("\n[1] Loading GRM...")

grm_raw = readdlm(GRM_FILE, ',', Float64, '\n', header=false)
grm_ids = Int.(grm_raw[:, 1])
G = grm_raw[:, 2:end]
n = size(G, 1)

@assert size(G, 1) == size(G, 2) "GRM is not square: $(size(G))"
println("  GRM: $n × $n individuals  (diagonal mean = $(round(mean(diag(G)), digits=4)))")

# =============================================================================
# 2. IDENTIFY LAST-GENERATION CANDIDATES AND LOAD POSTERIOR MEAN EBVs
# =============================================================================
# The QTL-MAS 2010 design has 3226 individuals across 5 generations.
# Generations 1-4 (n=2326) are phenotyped; generation 5 (n=900) is unphenotyped.
# Last-gen IDs = all IDs in tbv.txt that are NOT in phenotypes.txt.
# These animals have the least precise EBVs (no own phenotype) — the realistic scenario.
println("\n[2] Identifying last-generation candidates and loading posterior mean EBVs...")

# Load all IDs and phenotyped IDs from source files
tbv_raw   = readdlm(TBV_FILE,   ',', header=false)
pheno_raw = readdlm(PHENO_FILE, ',', header=false)

all_ids_tbv    = Int.(tbv_raw[:, 1])
phenotyped_ids = Set(Int.(pheno_raw[:, 1]))
lastgen_ids_set = Set(id for id in all_ids_tbv if id ∉ phenotyped_ids)

n_lastgen = length(lastgen_ids_set)
println("  Total individuals (tbv.txt)   : $(length(all_ids_tbv))")
println("  Phenotyped (phenotypes.txt)   : $(length(phenotyped_ids))")
println("  Last-generation candidates    : $n_lastgen  (unphenotyped)")
@assert n_lastgen == 900 "Expected 900 last-gen animals, found $n_lastgen — check tbv/pheno files"

# Restrict GRM to last-gen candidates only (preserving GRM row order)
lastgen_mask    = [id in lastgen_ids_set for id in grm_ids]
lastgen_indices = findall(lastgen_mask)     # row/col positions in full GRM
lastgen_ids     = grm_ids[lastgen_indices]  # ordered IDs for the 900×900 submatrix

n_cand = length(lastgen_ids)   # = 900
G = G[lastgen_indices, lastgen_indices]

println("  Sub-GRM for candidates        : $n_cand × $n_cand")
println("  Sub-GRM diagonal mean         : $(round(mean(diag(G)), digits=4))")

# Build sex indicator vectors aligned to lastgen_ids (GRM order)
# tbv.txt col 2: sex code — 1 = male (sire), 2 = female (dam)  [verify for your dataset]
id_to_sex = Dict(Int(tbv_raw[i, 1]) => Int(tbv_raw[i, 2]) for i in 1:size(tbv_raw, 1))
sex_vec   = [get(id_to_sex, id, 0) for id in lastgen_ids]

sire_idx = findall(sex_vec .== 0)   # indices of males among 900 candidates
dam_idx  = findall(sex_vec .== 1)   # indices of females among 900 candidates

n_sires = length(sire_idx)
n_dams  = length(dam_idx)
println("  Sires (males)  in last-gen    : $n_sires")
println("  Dams  (females) in last-gen   : $n_dams")
@assert n_sires > 0 "No sires found — check sex coding in tbv.txt col 2"
@assert n_dams  > 0 "No dams  found — check sex coding in tbv.txt col 2"

# Load JWAS posterior mean EBV files
# JWAS native format: space/tab-delimited with header; column 1 = ID, column 2 = EBV
# (column name varies by JWAS version — we grab by position to be robust)
means_y1_df = CSV.read(EBV_Y1_MEANS, DataFrame, delim=',')
means_y2_df = CSV.read(EBV_Y2_MEANS, DataFrame, delim=',')

# Use column positions: col 1 = ID, col 2 = posterior mean EBV
# Print column names so you can verify if something looks off
println("  EBV_y1 columns: $(names(means_y1_df))")
println("  EBV_y2 columns: $(names(means_y2_df))")

id_col_y1  = Symbol(names(means_y1_df)[1])
ebv_col_y1 = Symbol(names(means_y1_df)[2])
id_col_y2  = Symbol(names(means_y2_df)[1])
ebv_col_y2 = Symbol(names(means_y2_df)[2])

id_to_ebv_y1 = Dict(parse(Int, string(row[id_col_y1])) => Float64(row[ebv_col_y1])
                    for row in eachrow(means_y1_df))
id_to_ebv_y2 = Dict(parse(Int, string(row[id_col_y2])) => Float64(row[ebv_col_y2])
                    for row in eachrow(means_y2_df))

ebv_map_y1 = [get(id_to_ebv_y1, id, NaN) for id in lastgen_ids]
ebv_map_y2 = [get(id_to_ebv_y2, id, NaN) for id in lastgen_ids]

println("  Last-gen individuals with y1 EBV: $(sum(.!isnan.(ebv_map_y1)))")
println("  Last-gen individuals with y2 EBV: $(sum(.!isnan.(ebv_map_y2)))")

# Standardise each trait across the 900 candidates, then average into selection index
function standardise(v::Vector{Float64})
    valid = filter(!isnan, v)
    μ = mean(valid);  σ = std(valid)
    return (v .- μ) ./ σ
end

g_map = (standardise(ebv_map_y1) .+ standardise(ebv_map_y2)) ./ 2.0
replace!(g_map, NaN => 0.0)

println("  Index range (MAP): [$(round(minimum(g_map), digits=3)), $(round(maximum(g_map), digits=3))]")

# =============================================================================
# 3. LOAD MCMC CHAINS AND BUILD SCENARIO MATRIX (LAST-GEN ONLY)
# =============================================================================
# MCMC chain files layout (from JWASGBLUP_QTLMAS2010_9.jl):
#   rows    = MCMC iterations  (l samples post-burnin, thinned)
#   columns = individuals      (column header = individual ID as string)
# This is confirmed correct — no transposition needed.
println("\n[3] Loading MCMC chains and building scenario matrix for $n_cand candidates...")

chains_y1_df = CSV.read(EBV_Y1_CHAINS, DataFrame)
chains_y2_df = CSV.read(EBV_Y2_CHAINS, DataFrame)

chain_ids_y1 = parse.(Int, names(chains_y1_df))
chain_ids_y2 = parse.(Int, names(chains_y2_df))

mat_y1 = Matrix{Float64}(chains_y1_df)   # (l × n_all)
mat_y2 = Matrix{Float64}(chains_y2_df)

l_scenarios = size(mat_y1, 1)
@assert size(mat_y1) == size(mat_y2) "Chain dimensions differ between traits"
println("  MCMC scenarios (l)          : $l_scenarios")
println("  Individuals in chain files  : $(size(mat_y1, 2))")

# Map chain column index by individual ID
chain_id_to_col_y1 = Dict(chain_ids_y1[c] => c for c in eachindex(chain_ids_y1))
chain_id_to_col_y2 = Dict(chain_ids_y2[c] => c for c in eachindex(chain_ids_y2))

# Pre-extract chain columns for the 900 candidates only (avoids per-scenario lookup overhead)
# col_y1_cand[i] = column index in mat_y1 for candidate i   (missing → 0 placeholder)
col_y1_cand = [get(chain_id_to_col_y1, id, 0) for id in lastgen_ids]
col_y2_cand = [get(chain_id_to_col_y2, id, 0) for id in lastgen_ids]

missing_y1 = sum(col_y1_cand .== 0)
missing_y2 = sum(col_y2_cand .== 0)
missing_y1 > 0 && @warn "$missing_y1 last-gen individuals not found in y1 chain"
missing_y2 > 0 && @warn "$missing_y2 last-gen individuals not found in y2 chain"

# Build scenario matrix U of size (n_cand × l_scenarios)
# U[i, j] = standardised selection index for candidate i in MCMC iteration j
println("  Building U ($n_cand × $l_scenarios)...")

U = zeros(Float64, n_cand, l_scenarios)

for j in 1:l_scenarios
    # Extract raw EBVs for this iteration — candidates only
    ebv_j_y1 = [col_y1_cand[i] > 0 ? mat_y1[j, col_y1_cand[i]] : NaN for i in 1:n_cand]
    ebv_j_y2 = [col_y2_cand[i] > 0 ? mat_y2[j, col_y2_cand[i]] : NaN for i in 1:n_cand]

    # Standardise within iteration across candidates, then average traits
    idx_j = (standardise(ebv_j_y1) .+ standardise(ebv_j_y2)) ./ 2.0
    replace!(idx_j, NaN => 0.0)
    U[:, j] = idx_j
end

g_mean_scenarios = vec(mean(U, dims=2))
@printf("  Posterior mean index range (scenario mean): [%.3f, %.3f]\n",
        minimum(g_mean_scenarios), maximum(g_mean_scenarios))
@printf("  Correlation with MAP index (sanity check) : %.4f\n",
        cor(g_mean_scenarios, g_map))

# =============================================================================
# 4. COSMO SOLVER FACTORY
# =============================================================================

function make_cosmo_optimizer()
    return optimizer_with_attributes(
        COSMO.Optimizer,
        "max_iter" => COSMO_MAX_ITER,
        "eps_abs"  => COSMO_EPS_ABS,
        "eps_rel"  => COSMO_EPS_REL,
        "verbose"  => false
    )
end

# =============================================================================
# 5. MAP-OCS REFERENCE (posterior mean EBVs, μ = 0)
# =============================================================================

"""
    run_map_ocs(G, g, theta) -> (c, gain, coancestry, status)

Standard OCS using fixed (posterior mean) breeding values.
This is the μ=0 special case of CVaR-OCS.
"""
function run_map_ocs(G::Matrix{Float64}, g::Vector{Float64}, theta::Float64;
                    sire_idx::Vector{Int}=Int[], dam_idx::Vector{Int}=Int[])
    n = length(g)
    model = Model(make_cosmo_optimizer())

    @variable(model, c[1:n] >= 0)
    @constraint(model, sum(c) == 1.0)
    @constraint(model, 0.5 * c' * G * c <= theta)

    # Sex-specific contribution constraints: each sex contributes at most 0.5
    # Ensures equal sire/dam representation; prevents one sex dominating
    if !isempty(sire_idx)
        @constraint(model, sum(c[sire_idx]) <= 0.5)
    end
    if !isempty(dam_idx)
        @constraint(model, sum(c[dam_idx])  <= 0.5)
    end

    @objective(model, Max, dot(g, c))

    optimize!(model)
    status = termination_status(model)

    if status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        cv = value.(c)
        return cv, dot(g, cv), 0.5 * cv' * G * cv, status
    else
        @warn "MAP-OCS did not converge: $status"
        return zeros(n), NaN, NaN, status
    end
end

# =============================================================================
# 6. CVaR-OCS CORE SOLVER
# =============================================================================

"""
    run_cvar_ocs(G, U, theta, mu, alpha) -> (c, eta, gain_exp, cvar_gain, coancestry, status)

Solves the CVaR-aware OCS QP:

  max_{c, η, z}   (1/l) Σ_j c'u_j  +  μ [ η − 1/((1−α)l) Σ_j z_j ]
  s.t.
    z_j ≥ η − c'u_j   ∀j          (shortfall above VaR)
    z_j ≥ 0            ∀j
    (1/2) c'G c ≤ θ
    sum(c) = 1
    c ≥ 0

Arguments:
  G     — n×n genomic relationship matrix
  U     — n×l scenario matrix  (column j = index EBVs for MCMC draw j)
  theta — coancestry upper bound Θ
  mu    — risk-aversion weight (≥0; 0 → posterior-mean OCS)
  alpha — CVaR confidence level ∈ (0,1)

Returns:
  c           — optimal contribution vector (n)
  eta         — optimal VaR threshold
  gain_exp    — expected genetic gain across scenarios: (1/l) Σ_j c'u_j
  cvar_gain   — CVaR of genetic gain: η − (1/((1−α)l)) Σ_j z_j
  coancestry  — realised coancestry: (1/2) c'G c
  status      — JuMP termination status
"""
function run_cvar_ocs(G::Matrix{Float64}, U::Matrix{Float64},
                      theta::Float64, mu::Float64, alpha::Float64;
                      sire_idx::Vector{Int}=Int[], dam_idx::Vector{Int}=Int[])
    n, l = size(U)
    inv_alpha_l = 1.0 / ((1.0 - alpha) * l)

    model = Model(make_cosmo_optimizer())

    # Decision variables
    @variable(model, c[1:n] >= 0)          # contributions
    @variable(model, eta)                   # VaR threshold (free variable)
    @variable(model, z[1:l] >= 0)          # shortfall per scenario

    # Feasibility constraints
    @constraint(model, sum(c) == 1.0)
    @constraint(model, 0.5 * c' * G * c <= theta)

    # Sex-specific contribution constraints: each sex contributes at most 0.5
    if !isempty(sire_idx)
        @constraint(model, sum(c[sire_idx]) <= 0.5)
    end
    if !isempty(dam_idx)
        @constraint(model, sum(c[dam_idx])  <= 0.5)
    end

    # Shortfall constraints: z_j ≥ η − c'u_j  (binding whenever gain < η)
    # Written as: η − dot(U[:, j], c) − z[j] ≤ 0  for all j
    for j in 1:l
        @constraint(model, eta - dot(U[:, j], c) - z[j] <= 0)
    end

    # Objective: expected gain + μ × CVaR
    # (1/l) Σ_j c'u_j  =  c' * (U * ones(l) / l)  = dot(g_mean, c)
    g_bar = vec(mean(U, dims=2))   # posterior mean across scenarios
    cvar_term = eta - inv_alpha_l * sum(z)

    @objective(model, Max, dot(g_bar, c) + mu * cvar_term)

    optimize!(model)
    status = termination_status(model)

    if status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        cv   = value.(c)
        etav = value(eta)
        zv   = value.(z)

        gain_exp   = dot(g_bar, cv)
        cvar_gain  = etav - inv_alpha_l * sum(zv)
        coancestry = 0.5 * cv' * G * cv

        return cv, etav, gain_exp, cvar_gain, coancestry, status
    else
        @warn "CVaR-OCS did not converge (μ=$mu, α=$alpha): $status"
        return zeros(n), NaN, NaN, NaN, NaN, status
    end
end

# =============================================================================
# 7. HELPER: CONTRIBUTION DISTRIBUTION METRICS
# =============================================================================

"""
Compute contribution distribution metrics for a solution vector c.
Returns a NamedTuple with n_selected, mean_c, max_c, max_pct, gini.
"""
function contribution_metrics(c::Vector{Float64}, threshold::Float64=CONTRIBUTION_THRESHOLD)
    sel = c[c .> threshold]
    isempty(sel) && return (n_selected=0, mean_c=NaN, max_c=NaN, max_pct=NaN, gini=NaN)

    sorted = sort(sel)
    n_sel  = length(sorted)
    s_sum  = sum(sorted)
    gini   = sum((2i - n_sel - 1) * sorted[i] for i in 1:n_sel) / (n_sel * s_sum)

    return (
        n_selected = n_sel,
        mean_c     = mean(sorted),
        max_c      = maximum(sorted),
        max_pct    = maximum(sorted) / s_sum * 100.0,
        gini       = gini
    )
end

# =============================================================================
# 8. RUN MAP-OCS REFERENCE
# =============================================================================
println("\n[4] Running MAP-OCS reference solution (μ=0, posterior mean EBVs)...")

c_map, gain_map, coanc_map, status_map = run_map_ocs(G, g_map, THETA;
                                                    sire_idx=sire_idx, dam_idx=dam_idx)
m_map = contribution_metrics(c_map)

@printf("  Status       : %s\n", string(status_map))
@printf("  Expected gain: %.6f\n", gain_map)
@printf("  Coancestry   : %.6f  (limit = %.3f)\n", coanc_map, THETA)
@printf("  N selected   : %d\n", m_map.n_selected)
@printf("  Max contrib  : %.4f  (%.1f%% of total)\n", m_map.max_c, m_map.max_pct)
@printf("  Gini coeff   : %.4f\n", m_map.gini)

# Evaluate MAP solution against scenario distribution (in-sample)
map_scenario_gains = [dot(c_map, U[:, j]) for j in 1:l_scenarios]
@printf("  In-sample expected gain (scenario mean): %.6f\n", mean(map_scenario_gains))
@printf("  In-sample VaR(0.95)                    : %.6f\n", quantile(map_scenario_gains, 0.05))
@printf("  In-sample CVaR(0.95)                   : %.6f\n",
        mean(map_scenario_gains[map_scenario_gains .< quantile(map_scenario_gains, 0.05)]))

# =============================================================================
# 9. SWEEP OVER (μ, α) — BUILD RISK-RETURN FRONTIER
# =============================================================================
println("\n[5] Sweeping CVaR-OCS over μ and α...")
println("    α values : $(ALPHA_VALUES)")
println("    μ values : $(MU_VALUES)")

# Storage for frontier
frontier_rows = []
solutions     = Dict{Tuple{Float64,Float64}, Vector{Float64}}()

# Store MAP-OCS reference row
push!(frontier_rows, (
    label = "MAP-OCS", mu = 0.0, alpha = NaN,
    gain_exp = mean(map_scenario_gains), cvar_gain = NaN,
    coancestry = coanc_map,
    n_selected = m_map.n_selected,
    max_contrib = m_map.max_c, max_pct = m_map.max_pct, gini = m_map.gini,
    var95 = quantile(map_scenario_gains, 0.05),
    cvar95_eval = mean(map_scenario_gains[map_scenario_gains .< quantile(map_scenario_gains, 0.05)]),
    status = string(status_map)
))

for alpha in ALPHA_VALUES
    for mu in MU_VALUES
        # μ=0 means the CVaR term vanishes, leaving η unconstrained — the solver pushes
        # η to -∞ to trivially satisfy shortfall constraints, producing a meaningless CVaR.
        # MAP-OCS (already stored above) is the correct μ=0 reference; skip here.
        mu == 0.0 && continue

        label = @sprintf("CVaR(α=%.2f,μ=%.1f)", alpha, mu)
        print("  Running $label ...")

        c, eta, gain_exp, cvar_gain, coanc, status = run_cvar_ocs(G, U, THETA, mu, alpha;
                                                                    sire_idx=sire_idx, dam_idx=dam_idx)

        if status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
            m = contribution_metrics(c)
            sc_gains = [dot(c, U[:, j]) for j in 1:l_scenarios]
            var95    = quantile(sc_gains, 0.05)
            cvar95   = mean(sc_gains[sc_gains .< var95])

            push!(frontier_rows, (
                mu = mu, alpha = alpha, label = label,
                gain_exp = gain_exp, cvar_gain = cvar_gain,
                coancestry = coanc,
                n_selected = m.n_selected,
                max_contrib = m.max_c, max_pct = m.max_pct, gini = m.gini,
                var95 = var95, cvar95_eval = cvar95,
                status = string(status)
            ))
            solutions[(mu, alpha)] = c

            @printf("  gain=%.4f  CVaR=%.4f  n_sel=%d  gini=%.3f\n",
                    gain_exp, cvar_gain, m.n_selected, m.gini)
        else
            @printf("  FAILED (%s)\n", string(status))
        end
    end
end

# =============================================================================
# 10. SAVE FRONTIER TABLE
# =============================================================================
println("\n[6] Saving results...")

mkpath(joinpath(SAVE_DIR, theta_folder))

frontier_df = DataFrame(frontier_rows)
frontier_file = joinpath(SAVE_DIR, theta_folder, "cvar_ocs_frontier.csv")
CSV.write(frontier_file, frontier_df)
println("  Frontier table → $frontier_file")

# =============================================================================
# 11. SAVE FULL CONTRIBUTION VECTORS
# =============================================================================

# Build wide DataFrame: columns = individual IDs; rows = MAP-OCS + each (μ,α) run
contrib_df = DataFrame(
    label     = String[],
    mu        = Float64[],
    alpha     = Float64[],
)
for id in lastgen_ids
    contrib_df[!, "ID_$id"] = Float64[]
end

# MAP row
map_row = vcat(["MAP-OCS", 0.0, NaN], c_map)
push!(contrib_df, map_row)

for ((mu, alpha), c) in solutions
    label = @sprintf("CVaR_a%.2f_mu%.1f", alpha, mu)
    row   = vcat([label, mu, alpha], c)
    push!(contrib_df, row)
end

contrib_file = joinpath(SAVE_DIR, theta_folder, "cvar_ocs_solutions.csv")
CSV.write(contrib_file, contrib_df)
println("  Contribution vectors → $contrib_file")

# =============================================================================
# 12. SUMMARY REPORT
# =============================================================================

summary_file = joinpath(SAVE_DIR, theta_folder, "cvar_ocs_summary.txt")
open(summary_file, "w") do io
    println(io, "=" ^ 70)
    println(io, "CVaR-OCS SUMMARY — QTL-MAS 2010")
    println(io, "=" ^ 70)
    println(io, "Θ (coancestry limit) = $THETA")
    println(io, "l (scenarios)        = $l_scenarios")
    println(io, "n (candidates)       = $n_cand  (last generation only, unphenotyped)")
    println(io, "")
    println(io, "─" ^ 70)
    @printf(io, "%-28s %8s %8s %8s %6s %7s\n",
            "Model", "E[gain]", "CVaR95", "Coancest", "N_sel", "Gini")
    println(io, "─" ^ 70)
    for row in eachrow(frontier_df)
        @printf(io, "%-28s %8.4f %8.4f %8.5f %6d %7.4f\n",
                row.label,
                row.gain_exp,
                isnan(row.cvar95_eval) ? 0.0 : row.cvar95_eval,
                row.coancestry,
                row.n_selected,
                isnan(row.gini) ? 0.0 : row.gini)
    end
    println(io, "─" ^ 70)
    println(io, """
Notes:
  E[gain]   = (1/l) Σ_j c'g^(j)  — expected genetic gain across MCMC scenarios
  CVaR95    = E[gain | gain < VaR(0.05)]  — mean gain in worst 5% of scenarios
  Coancest  = (1/2) c'Σ c         — realised group coancestry
  N_sel     = number of individuals with c > 1e-4
  Gini      = Gini coefficient of contribution distribution (0=equal, 1=concentrated)

Parameters:
  μ  (mu)   — risk-aversion weight: higher → more weight on CVaR tail protection
  α (alpha) — CVaR confidence: α=0.95 means protect against worst (1−0.95)=5% of scenarios
  μ=0       → posterior-mean MAP-OCS (no tail risk adjustment)
  μ→∞, α→1  → approaches minimax robust OCS (Fogg et al. 2024)
""")
end
println("  Summary → $summary_file")

# =============================================================================
# 13. COMPARE MAP-OCS vs SELECTED CVaR-OCS
# =============================================================================
println("\n" * "=" ^ 70)
println("COMPARISON: MAP-OCS vs CVaR-OCS")
println("=" ^ 70)

if haskey(solutions, (MU_VALUES[end], ALPHA_VALUES[2]))
    c_cvar = solutions[(MU_VALUES[end], ALPHA_VALUES[2])]
    sc_gains_cvar = [dot(c_cvar, U[:, j]) for j in 1:l_scenarios]
    m_cvar = contribution_metrics(c_cvar)

    @printf("\n  %-30s  %8s  %8s  %6s  %7s\n", "Model", "E[gain]", "CVaR95", "N_sel", "Gini")
    @printf("  %-30s  %8.4f  %8.4f  %6d  %7.4f\n",
            "MAP-OCS",
            mean(map_scenario_gains),
            mean(map_scenario_gains[map_scenario_gains .< quantile(map_scenario_gains, 0.05)]),
            m_map.n_selected, m_map.gini)

    @printf("  %-30s  %8.4f  %8.4f  %6d  %7.4f\n",
            @sprintf("CVaR-OCS (α=%.2f, μ=%.1f)", ALPHA_VALUES[2], MU_VALUES[end]),
            mean(sc_gains_cvar),
            mean(sc_gains_cvar[sc_gains_cvar .< quantile(sc_gains_cvar, 0.05)]),
            m_cvar.n_selected, m_cvar.gini)

    delta_gain = (mean(sc_gains_cvar) - mean(map_scenario_gains)) / abs(mean(map_scenario_gains)) * 100.0
    delta_cvar = (mean(sc_gains_cvar[sc_gains_cvar .< quantile(sc_gains_cvar, 0.05)]) -
                  mean(map_scenario_gains[map_scenario_gains .< quantile(map_scenario_gains, 0.05)])) /
                 abs(mean(map_scenario_gains[map_scenario_gains .< quantile(map_scenario_gains, 0.05)])) * 100.0

    @printf("\n  Expected gain change : %+.2f%%\n", delta_gain)
    @printf("  CVaR95 gain change   : %+.2f%%\n", delta_cvar)
end


# =============================================================================
# ORACLE VALIDATION — MAP-OCS on True Breeding Values
# =============================================================================
# Reviewers requested simulation-based validation using known TBVs.
# We run MAP-OCS using TBVs as the objective vector (oracle / gold standard),
# then evaluate ALL three solutions (MAP-EBV, CVaR-OCS, MAP-TBV) against TBVs.
#
# Three key comparisons:
#   1. Realised TBV gain  : c' * tbv_index  for each solution
#   2. Jaccard overlap    : pairwise set similarity of selected individuals
#   3. Gap closure        : does CVaR-OCS recover some of the MAP-EBV → MAP-TBV gap?
# =============================================================================

println("\n" * "=" ^ 70)
println("ORACLE VALIDATION — MAP-OCS on True Breeding Values")
println("=" ^ 70)

# ── Load TBVs for last-gen candidates ────────────────────────────────────────
println("\n[V1] Loading TBVs for last-generation candidates...")

tbv_raw_v   = readdlm(TBV_FILE,   ',', header=false)
pheno_raw_v = readdlm(PHENO_FILE, ',', header=false)

all_ids_v      = Int.(tbv_raw_v[:, 1])
tbv_y1_all_v   = Float64.(tbv_raw_v[:, 5])    # col 5  = TBV trait 1 (Q)
tbv_y2_all_v   = Float64.(tbv_raw_v[:, 13])   # col 13 = TBV trait 2 (B)
phenotyped_v   = Set(Int.(pheno_raw_v[:, 1]))

# Build TBV vector aligned to lastgen_ids (GRM order) used throughout the script
id_to_tbv_y1 = Dict(all_ids_v[i] => tbv_y1_all_v[i] for i in eachindex(all_ids_v))
id_to_tbv_y2 = Dict(all_ids_v[i] => tbv_y2_all_v[i] for i in eachindex(all_ids_v))

tbv_y1_cand = [get(id_to_tbv_y1, id, NaN) for id in lastgen_ids]
tbv_y2_cand = [get(id_to_tbv_y2, id, NaN) for id in lastgen_ids]

function standardise_v(v::Vector{Float64})
    valid = filter(!isnan, v)
    return (v .- mean(valid)) ./ std(valid)
end

# Selection index on TBVs — same standardise-then-average convention as EBV index
tbv_index = (standardise_v(tbv_y1_cand) .+ standardise_v(tbv_y2_cand)) ./ 2.0
replace!(tbv_index, NaN => 0.0)

@printf("  TBV index range: [%.3f, %.3f]\n", minimum(tbv_index), maximum(tbv_index))
@printf("  Correlation EBV index vs TBV index: %.4f\n", cor(g_map, tbv_index))

# ── Run MAP-OCS on TBVs (oracle) ─────────────────────────────────────────────
println("\n[V2] Running MAP-OCS(TBV) — oracle solution...")

c_oracle, gain_oracle_obj, coanc_oracle, status_oracle = run_map_ocs(G, tbv_index, THETA;
                                                                   sire_idx=sire_idx, dam_idx=dam_idx)
m_oracle = contribution_metrics(c_oracle)

@printf("  Status          : %s\n", string(status_oracle))
@printf("  Objective gain  : %.6f  (optimised against TBVs)\n", gain_oracle_obj)
@printf("  Coancestry      : %.6f  (limit = %.3f)\n", coanc_oracle, THETA)
@printf("  N selected      : %d\n", m_oracle.n_selected)
@printf("  Gini            : %.4f\n", m_oracle.gini)

# ── Evaluate ALL solutions against TBVs ──────────────────────────────────────
println("\n[V3] Evaluating all solutions against TBVs...")

# Collect best CVaR solution per α (highest cvar95_eval, already in frontier_rows)
# Re-derive from solutions dict: best μ per α is the one with highest in-sample CVaR95
function best_cvar_solution(alpha_val::Float64)
    candidates = [(mu, alpha) for (mu, alpha) in keys(solutions) if abs(alpha - alpha_val) < 0.001]
    isempty(candidates) && return nothing, nothing
    # Evaluate each against scenario matrix to find best CVaR95
    best_key = nothing
    best_cvar95 = -Inf
    for key in candidates
        c = solutions[key]
        sc = [dot(c, U[:, j]) for j in 1:l_scenarios]
        cvar95 = mean(sc[sc .< quantile(sc, 0.05)])
        if cvar95 > best_cvar95
            best_cvar95 = cvar95
            best_key = key
        end
    end
    return best_key, solutions[best_key]
end

key_090, c_best_090 = best_cvar_solution(0.90)
key_095, c_best_095 = best_cvar_solution(0.95)
key_099, c_best_099 = best_cvar_solution(0.99)

# Realised TBV gain = c' * tbv_index  for each solution
realised_tbv(c::Vector{Float64}) = dot(c, tbv_index)

gain_tbv_map     = realised_tbv(c_map)
gain_tbv_oracle  = realised_tbv(c_oracle)
gain_tbv_090     = c_best_090 !== nothing ? realised_tbv(c_best_090) : NaN
gain_tbv_095     = c_best_095 !== nothing ? realised_tbv(c_best_095) : NaN
gain_tbv_099     = c_best_099 !== nothing ? realised_tbv(c_best_099) : NaN

# Gap closure: how much of the MAP-EBV → Oracle gap does each CVaR solution recover?
# Gap closure % = (CVaR gain - MAP-EBV gain) / (Oracle gain - MAP-EBV gain) * 100
gap = gain_tbv_oracle - gain_tbv_map
gap_closure(g) = isnan(g) ? NaN : (g - gain_tbv_map) / gap * 100.0

println("\n  Realised TBV gain (c\'*tbv_index):")
@printf("  %-32s  %8.4f  (gap closure: reference)\n", "MAP-OCS(EBV)", gain_tbv_map)
@printf("  %-32s  %8.4f  (gap closure: +100%% — oracle)\n", "MAP-OCS(TBV) [oracle]", gain_tbv_oracle)
@printf("  %-32s  %8.4f  (gap closure: %+.1f%%)\n",
        @sprintf("CVaR-OCS(α=0.90,μ=%.1f)", key_090[1]), gain_tbv_090, gap_closure(gain_tbv_090))
@printf("  %-32s  %8.4f  (gap closure: %+.1f%%)\n",
        @sprintf("CVaR-OCS(α=0.95,μ=%.1f)", key_095[1]), gain_tbv_095, gap_closure(gain_tbv_095))
@printf("  %-32s  %8.4f  (gap closure: %+.1f%%)\n",
        @sprintf("CVaR-OCS(α=0.99,μ=%.1f)", key_099[1]), gain_tbv_099, gap_closure(gain_tbv_099))

# ── Jaccard overlap ───────────────────────────────────────────────────────────
println("\n[V4] Jaccard overlap between selected sets...")

function selected_set(c::Vector{Float64})
    return Set(lastgen_ids[c .> CONTRIBUTION_THRESHOLD])
end

function jaccard(a::Set, b::Set)
    isempty(a) && isempty(b) && return 1.0
    return length(intersect(a, b)) / length(union(a, b))
end

sel_map    = selected_set(c_map)
sel_oracle = selected_set(c_oracle)
sel_090    = c_best_090 !== nothing ? selected_set(c_best_090) : Set{Int}()
sel_095    = c_best_095 !== nothing ? selected_set(c_best_095) : Set{Int}()
sel_099    = c_best_099 !== nothing ? selected_set(c_best_099) : Set{Int}()

println("\n  Pairwise Jaccard similarity (higher = more overlap):")
@printf("  %-32s  vs  %-32s  :  %.3f\n",
        "MAP-OCS(EBV)", "MAP-OCS(TBV) [oracle]", jaccard(sel_map, sel_oracle))
@printf("  %-32s  vs  %-32s  :  %.3f\n",
        "CVaR-OCS(α=0.90)", "MAP-OCS(TBV) [oracle]", jaccard(sel_090, sel_oracle))
@printf("  %-32s  vs  %-32s  :  %.3f\n",
        "CVaR-OCS(α=0.95)", "MAP-OCS(TBV) [oracle]", jaccard(sel_095, sel_oracle))
@printf("  %-32s  vs  %-32s  :  %.3f\n",
        "CVaR-OCS(α=0.99)", "MAP-OCS(TBV) [oracle]", jaccard(sel_099, sel_oracle))
@printf("  %-32s  vs  %-32s  :  %.3f\n",
        "MAP-OCS(EBV)", "CVaR-OCS(α=0.95)",  jaccard(sel_map, sel_095))

println("\n  Overlap counts vs oracle (N in oracle=$(length(sel_oracle))):")
for (label, sel) in [("MAP-OCS(EBV)", sel_map),
                      ("CVaR-OCS(α=0.90)", sel_090),
                      ("CVaR-OCS(α=0.95)", sel_095),
                      ("CVaR-OCS(α=0.99)", sel_099)]
    shared   = length(intersect(sel, sel_oracle))
    only_me  = length(setdiff(sel, sel_oracle))
    only_orc = length(setdiff(sel_oracle, sel))
    @printf("  %-22s  shared=%d  unique_to_me=%d  oracle_only=%d\n",
            label, shared, only_me, only_orc)
end

# ── Save oracle validation results ───────────────────────────────────────────
println("\n[V5] Saving oracle validation results...")

oracle_df = DataFrame(
    model          = ["MAP-OCS(EBV)", "MAP-OCS(TBV)",
                      @sprintf("CVaR-OCS(a0.90,mu%.1f)", key_090 !== nothing ? key_090[1] : 0.0),
                      @sprintf("CVaR-OCS(a0.95,mu%.1f)", key_095 !== nothing ? key_095[1] : 0.0),
                      @sprintf("CVaR-OCS(a0.99,mu%.1f)", key_099 !== nothing ? key_099[1] : 0.0)],
    realised_tbv_gain = [gain_tbv_map, gain_tbv_oracle,
                         gain_tbv_090, gain_tbv_095, gain_tbv_099],
    gap_closure_pct   = [0.0, 100.0,
                         gap_closure(gain_tbv_090),
                         gap_closure(gain_tbv_095),
                         gap_closure(gain_tbv_099)],
    n_selected        = [m_map.n_selected, m_oracle.n_selected,
                         c_best_090 !== nothing ? contribution_metrics(c_best_090).n_selected : 0,
                         c_best_095 !== nothing ? contribution_metrics(c_best_095).n_selected : 0,
                         c_best_099 !== nothing ? contribution_metrics(c_best_099).n_selected : 0],
    jaccard_vs_oracle = [jaccard(sel_map, sel_oracle), 1.0,
                         jaccard(sel_090, sel_oracle),
                         jaccard(sel_095, sel_oracle),
                         jaccard(sel_099, sel_oracle)],
    coancestry        = [coanc_map, coanc_oracle, NaN, NaN, NaN]
)

# Add contribution vectors for oracle to solutions file
# Append as a new row
sol_df_oracle = CSV.read(contrib_file, DataFrame)
oracle_row_vals = vcat(["MAP-OCS(TBV)", 0.0, NaN], c_oracle)
push!(sol_df_oracle, oracle_row_vals)
CSV.write(contrib_file, sol_df_oracle)
println("  Oracle contributions appended to: $contrib_file")

oracle_file = joinpath(SAVE_DIR, theta_folder, "oracle_validation.csv")
CSV.write(oracle_file, oracle_df)
println("  Oracle validation table → $oracle_file")

println("\n" * "=" ^ 70)
println("ORACLE VALIDATION COMPLETE")
println("=" ^ 70)

println("\n" * "=" ^ 70)
println("CVaR-OCS COMPLETE")
println("=" ^ 70)
@printf("Outputs:\n  %s\n  %s\n  %s\n  %s\n",
        frontier_file, contrib_file, summary_file, oracle_file)
