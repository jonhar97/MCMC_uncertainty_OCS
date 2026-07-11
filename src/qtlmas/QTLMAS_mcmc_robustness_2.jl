"""
MCMC Robustness Analysis — QTL-MAS 2010 Simulation Study
=========================================================

Calculates robustness scores for individuals selected by MAP-OCS and/or
CVaR-OCS, then compares the two solutions across all MCMC scenarios.

Key differences from Norway spruce script:
  - Restricted to 900 last-generation unphenotyped candidates (sub-GRM)
  - Candidates = union(MAP-OCS selected, CVaR-OCS selected) — no replacement pool
  - CVaR-OCS contribution vector loaded from cvar_ocs_solutions.csv
  - Sex constraints (sire/dam ≤ 0.5) applied consistently
  - No constrained-OCS step (CVaR-OCS is the alternative solution)
  - 4-panel publication figure targeting the robustness × selection story

Author: Jon Ahlinder (Skogforsk) / Ahlinder & Waldmann
Date:   2026
"""

# Guard against duplicate loading warnings
if !@isdefined(QTLMAS_ROBUSTNESS_LOADED)
    global QTLMAS_ROBUSTNESS_LOADED = true
    println("Loading QTL-MAS Robustness Analysis...")
else
    println("Reloading QTL-MAS Robustness Analysis (functions will be redefined)...")
end

using DataFrames, CSV, Statistics, LinearAlgebra
using COSMO, JuMP, Plots, StatsPlots
using DelimitedFiles, Measures, JLD2, FileIO
using HypothesisTests, Distributions, Printf, Random

# =============================================================================
# CONFIGURATION
# =============================================================================

THETA              = 0.03    # Coancestry constraint — match CVaR-OCS run
SELECTION_THRESHOLD = 1e-4   # Contribution threshold for "selected"
N_ROBUSTNESS_SAMPLES = 200   # MCMC iterations used for robustness scoring
                              # (subset for speed; all used for variance eval)

# Which CVaR-OCS model to load as the alternative solution
# Must match a label in cvar_ocs_solutions.csv
CVAR_LABEL = "CVaR_a0.90_mu1.2"   # best model at Theta=0.05

BASE_DIR     = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\QTLMAS\\"
SAVE_DIR     = joinpath(BASE_DIR, "Save5") #joinpath(BASE_DIR, "Save6", "Theta5")   # folder for this Theta run
SAVE_DIR2    = joinpath(BASE_DIR, "Save5")  # alternative save directory
FIGURES_DIR  = joinpath(BASE_DIR, "Figures")

# Input files
GRM_FILE       = joinpath(BASE_DIR, "Save", "GRM_QTLMAS_jwas.txt")
TBV_FILE       = joinpath(BASE_DIR, "tbv.txt")
PHENO_FILE     = joinpath(BASE_DIR, "phenotypes.txt")
EBV_Y1_FILE    = joinpath(SAVE_DIR2, "EBV_y1.txt")
EBV_Y2_FILE    = joinpath(SAVE_DIR2, "EBV_y2.txt")
MCMC_Y1_FILE   = joinpath(SAVE_DIR2, "MCMC_samples_EBV_y1.txt")
MCMC_Y2_FILE   = joinpath(SAVE_DIR2, "MCMC_samples_EBV_y2.txt")
SOLUTIONS_FILE = joinpath(SAVE_DIR, "cvar_ocs_solutions.csv")
ORACLE_FILE    = joinpath(SAVE_DIR, "oracle_validation.csv")

# Colour palette (consistent with CVaR-OCS visualisation script)
const C_MAP    = RGB(0.20, 0.40, 0.65)   # blue   — MAP-OCS
const C_CVAR   = RGB(0.17, 0.63, 0.17)   # green  — CVaR-OCS
const C_ORACLE = RGB(0.50, 0.15, 0.65)   # purple — Oracle
const C_UNSEL  = RGB(0.75, 0.75, 0.75)   # grey   — unselected

# =============================================================================
# OCS FUNCTION WITH SEX CONSTRAINTS
# =============================================================================

"""
    run_ocs(G, g, sire_idx, dam_idx, theta) -> Vector{Float64}

MAP-OCS with coancestry constraint and sex-specific contribution limits.
Sex coding for QTL-MAS: 1 = male (sire), 0 = female (dam).
"""
function run_ocs(G::Matrix{Float64}, g::Vector{Float64},
                 sire_idx::Vector{Int}, dam_idx::Vector{Int},
                 theta::Float64)
    n = length(g)
    model = Model(optimizer_with_attributes(COSMO.Optimizer,
                  "max_iter" => 100000, "eps_abs" => 1e-4,
                  "eps_rel" => 1e-4, "verbose" => false))

    @variable(model, c[1:n] >= 0)
    @constraint(model, sum(c) == 1.0)
    @constraint(model, 0.5 * c' * G * c <= theta)
    @constraint(model, sum(c[sire_idx]) <= 0.5)
    @constraint(model, sum(c[dam_idx])  <= 0.5)
    @objective(model, Max, dot(g, c))

    optimize!(model)

    if termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        return value.(c)
    else
        @warn "OCS did not converge: $(termination_status(model))"
        return zeros(n)
    end
end

# =============================================================================
# DATA LOADING
# =============================================================================

"""
    load_data() -> NamedTuple

Load all inputs and return a consistent named tuple.
Applies last-gen subsetting, ID alignment, and sex vector construction.
"""
function load_data()
    println("=" ^ 70)
    println("LOADING DATA")
    println("=" ^ 70)

    # ── 1. Full GRM ──────────────────────────────────────────────────────────
    println("\n[1] Loading GRM...")
    grm_raw = readdlm(GRM_FILE, ',', Float64, '\n', header=false)
    grm_ids_full = Int.(grm_raw[:, 1])
    G_full = grm_raw[:, 2:end]
    println("  Full GRM: $(size(G_full, 1)) × $(size(G_full, 2))")

    # ── 2. Identify last-gen candidates (same logic as CVaR-OCS script) ──────
    println("\n[2] Identifying last-generation candidates...")
    tbv_raw   = readdlm(TBV_FILE,   ',', header=false)
    pheno_raw = readdlm(PHENO_FILE, ',', header=false)

    all_ids      = Int.(tbv_raw[:, 1])
    tbv_y1_all   = Float64.(tbv_raw[:, 5])    # col 5  = TBV trait 1
    tbv_y2_all   = Float64.(tbv_raw[:, 13])   # col 13 = TBV trait 2
    sex_all      = Int.(tbv_raw[:, 2])         # col 2  = sex (1=sire, 0=dam)
    phenotyped   = Set(Int.(pheno_raw[:, 1]))

    lastgen_set  = Set(id for id in all_ids if id ∉ phenotyped)
    n_cand       = length(lastgen_set)
    @assert n_cand == 900 "Expected 900 last-gen candidates, found $n_cand"
    println("  Last-gen candidates: $n_cand")

    # ── 3. Sub-GRM aligned to GRM row order ──────────────────────────────────
    lastgen_mask    = [id in lastgen_set for id in grm_ids_full]
    lastgen_indices = findall(lastgen_mask)
    lastgen_ids     = grm_ids_full[lastgen_indices]
    G               = G_full[lastgen_indices, lastgen_indices]
    n               = length(lastgen_ids)
    println("  Sub-GRM: $n × $n  (diagonal mean = $(round(mean(diag(G)), digits=4)))")

    # ── 4. Sex vectors (QTL-MAS: 1=sire, 0=dam) ─────────────────────────────
    id_to_sex = Dict(all_ids[i] => sex_all[i] for i in eachindex(all_ids))
    sex_vec   = [get(id_to_sex, id, -1) for id in lastgen_ids]
    sire_idx  = findall(sex_vec .== 1)
    dam_idx   = findall(sex_vec .== 0)
    println("  Sires (sex==1): $(length(sire_idx))  Dams (sex==0): $(length(dam_idx))")
    @assert length(sire_idx) > 0 && length(dam_idx) > 0 "Sex coding error — check tbv.txt col 2"

    # ── 5. TBV index (standardised) ──────────────────────────────────────────
    id_to_tbv_y1 = Dict(all_ids[i] => tbv_y1_all[i] for i in eachindex(all_ids))
    id_to_tbv_y2 = Dict(all_ids[i] => tbv_y2_all[i] for i in eachindex(all_ids))
    tbv_y1 = [get(id_to_tbv_y1, id, NaN) for id in lastgen_ids]
    tbv_y2 = [get(id_to_tbv_y2, id, NaN) for id in lastgen_ids]
    function stdise(v)
        valid = filter(!isnan, v)
        return (v .- mean(valid)) ./ std(valid)
    end
    tbv_index = (stdise(tbv_y1) .+ stdise(tbv_y2)) ./ 2.0
    replace!(tbv_index, NaN => 0.0)

    # ── 6. MAP posterior-mean EBV index ──────────────────────────────────────
    println("\n[3] Loading MAP EBVs...")
    ebv_y1_df = CSV.read(EBV_Y1_FILE, DataFrame, delim=',')
    ebv_y2_df = CSV.read(EBV_Y2_FILE, DataFrame, delim=',')

    id_col1 = Symbol(names(ebv_y1_df)[1]);  ebv_col1 = Symbol(names(ebv_y1_df)[2])
    id_col2 = Symbol(names(ebv_y2_df)[1]);  ebv_col2 = Symbol(names(ebv_y2_df)[2])

    id_to_ebv_y1 = Dict(parse(Int, string(r[id_col1])) => Float64(r[ebv_col1])
                         for r in eachrow(ebv_y1_df))
    id_to_ebv_y2 = Dict(parse(Int, string(r[id_col2])) => Float64(r[ebv_col2])
                         for r in eachrow(ebv_y2_df))

    ebv_y1 = [get(id_to_ebv_y1, id, NaN) for id in lastgen_ids]
    ebv_y2 = [get(id_to_ebv_y2, id, NaN) for id in lastgen_ids]
    g_map  = (stdise(ebv_y1) .+ stdise(ebv_y2)) ./ 2.0
    replace!(g_map, NaN => 0.0)
    println("  MAP index range: [$(round(minimum(g_map),digits=3)), $(round(maximum(g_map),digits=3))]")
    @printf("  Correlation EBV vs TBV: %.4f\n", cor(g_map, tbv_index))

    # ── 7. MCMC chains (900 × l matrix, candidates only) ─────────────────────
    println("\n[4] Loading MCMC chains...")
    cache_file = joinpath(SAVE_DIR, "mcmc_index_900_cache.jld2")

    if isfile(cache_file)
        println("  Loading from cache...")
        @load cache_file mcmc_index_900
        println("  ✓ $(size(mcmc_index_900,1)) iterations × $(size(mcmc_index_900,2)) candidates")
    else
        chains_y1 = CSV.read(MCMC_Y1_FILE, DataFrame, delim=',')
        chains_y2 = CSV.read(MCMC_Y2_FILE, DataFrame, delim=',')

        chain_ids_y1 = parse.(Int, names(chains_y1))
        chain_ids_y2 = parse.(Int, names(chains_y2))
        mat_y1 = Matrix{Float64}(chains_y1)
        mat_y2 = Matrix{Float64}(chains_y2)

        l = size(mat_y1, 1)
        println("  Chain dimensions: $l iterations × $(size(mat_y1,2)) individuals")

        col_y1 = Dict(chain_ids_y1[c] => c for c in eachindex(chain_ids_y1))
        col_y2 = Dict(chain_ids_y2[c] => c for c in eachindex(chain_ids_y2))
        cols_y1_cand = [get(col_y1, id, 0) for id in lastgen_ids]
        cols_y2_cand = [get(col_y2, id, 0) for id in lastgen_ids]

        println("  Building 900×$l scenario matrix...")
        mcmc_index_900 = zeros(Float64, l, n)   # rows=iterations, cols=candidates

        for j in 1:l
            ev1 = [cols_y1_cand[i] > 0 ? mat_y1[j, cols_y1_cand[i]] : NaN for i in 1:n]
            ev2 = [cols_y2_cand[i] > 0 ? mat_y2[j, cols_y2_cand[i]] : NaN for i in 1:n]
            idx = (stdise(ev1) .+ stdise(ev2)) ./ 2.0
            replace!(idx, NaN => 0.0)
            mcmc_index_900[j, :] = idx
        end

        @save cache_file mcmc_index_900
        println("  ✓ Cache saved: $cache_file")
    end

    # ── 8. Load CVaR-OCS contribution vector ─────────────────────────────────
    println("\n[5] Loading CVaR-OCS contributions ($CVAR_LABEL)...")
    sol_df   = CSV.read(SOLUTIONS_FILE, DataFrame)
    id_cols  = names(sol_df)[4:end]
    cvar_row = sol_df[String.(sol_df.label) .== CVAR_LABEL, :]

    if isempty(cvar_row)
        error("Label '$CVAR_LABEL' not found in $SOLUTIONS_FILE\n" *
              "Available: $(unique(String.(sol_df.label)))")
    end

    # Align solution columns to lastgen_ids order
    sol_id_lookup = Dict(parse(Int, replace(col, "ID_"=>"")) => col for col in id_cols)
    c_cvar = [haskey(sol_id_lookup, id) ? Float64(cvar_row[1, sol_id_lookup[id]]) : 0.0
              for id in lastgen_ids]

    # Load MAP-OCS contribution vector (label = "MAP-OCS")
    map_row = sol_df[String.(sol_df.label) .== "MAP-OCS", :]
    c_map_loaded = [haskey(sol_id_lookup, id) ? Float64(map_row[1, sol_id_lookup[id]]) : 0.0
                    for id in lastgen_ids]

    # Load oracle contribution vector
    oracle_row = sol_df[String.(sol_df.label) .== "MAP-OCS(TBV)", :]
    c_oracle = isempty(oracle_row) ? zeros(n) :
               [haskey(sol_id_lookup, id) ? Float64(oracle_row[1, sol_id_lookup[id]]) : 0.0
                for id in lastgen_ids]

    n_map  = sum(c_map_loaded  .> SELECTION_THRESHOLD)
    n_cvar = sum(c_cvar        .> SELECTION_THRESHOLD)
    n_orc  = sum(c_oracle      .> SELECTION_THRESHOLD)
    println("  MAP-OCS selected   : $n_map")
    println("  CVaR-OCS selected  : $n_cvar")
    println("  Oracle selected    : $n_orc")

    return (
        G            = G,
        n            = n,
        lastgen_ids  = lastgen_ids,
        sire_idx     = sire_idx,
        dam_idx      = dam_idx,
        g_map        = g_map,
        tbv_index    = tbv_index,
        mcmc_matrix  = mcmc_index_900,   # (l × n), rows=iterations
        c_map        = c_map_loaded,
        c_cvar       = c_cvar,
        c_oracle     = c_oracle,
        sex_vec      = sex_vec
    )
end

# =============================================================================
# ROBUSTNESS SCORE CALCULATION
# =============================================================================

"""
    run_ocs_exclude(G, g, sire_idx, dam_idx, theta, exclude_ind) -> Vector{Float64}

OCS with hard constraint c[exclude_ind] = 0.
All other candidates remain available — the solver redistributes optimally.
"""
function run_ocs_exclude(G::Matrix{Float64}, g::Vector{Float64},
                         sire_idx::Vector{Int}, dam_idx::Vector{Int},
                         theta::Float64, exclude_ind::Int)
    n = length(g)
    model = Model(optimizer_with_attributes(COSMO.Optimizer,
                  "max_iter" => 100000, "eps_abs" => 1e-4,
                  "eps_rel" => 1e-4, "verbose" => false))

    @variable(model, c[1:n] >= 0)
    @constraint(model, c[exclude_ind] == 0.0)   # hard exclusion
    @constraint(model, sum(c) == 1.0)
    @constraint(model, 0.5 * c' * G * c <= theta)
    if !isempty(sire_idx)
        @constraint(model, sum(c[sire_idx]) <= 0.5)
    end
    if !isempty(dam_idx)
        @constraint(model, sum(c[dam_idx])  <= 0.5)
    end
    @objective(model, Max, dot(g, c))

    optimize!(model)
    if termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        return value.(c)
    else
        @warn "run_ocs_exclude did not converge for ind=$exclude_ind"
        return zeros(n)
    end
end

"""
    calculate_robustness_scores(d, candidate_indices; n_samples) -> DataFrame

Robustness score = mean gain loss when individual i is excluded from the
full 900-candidate OCS, using a hard constraint c_i = 0.

Method (constrained exclusion over full candidate pool):
  For each candidate i and each MCMC iteration j:
    1. Run OCS on all 900 candidates → baseline gain G_base,j
       (individual i is free to be selected or not)
    2. Re-run OCS on all 900 candidates with CONSTRAINT c_i = 0
       (individual i is explicitly excluded)
    3. S_i,j = G_base,j - G_constrained,j
    4. S_i = mean_j(S_i,j)   [always ≥ 0 if i was selected in baseline]

Using c_i = 0 as a hard constraint (rather than degrading BV) ensures
the modified solution is genuinely optimal without i, making the
gain difference a true measure of i's contribution.

Note: for individuals NOT selected in the baseline, S_i ≈ 0 since
constraining a non-selected individual has no effect on the solution.
"""
function calculate_robustness_scores(d::NamedTuple,
                                     candidate_indices::Vector{Int};
                                     n_samples::Int = N_ROBUSTNESS_SAMPLES)
    println("\n" * "=" ^ 70)
    println("ROBUSTNESS SCORE CALCULATION (constrained exclusion)")
    println("=" ^ 70)
    println("  Method: hard constraint c_i=0, OCS re-run on all 900 candidates")
    println("  Scores ≥ 0: gain loss from excluding individual i")

    n_cand  = length(candidate_indices)
    l_total = size(d.mcmc_matrix, 1)
    n_use   = min(n_samples, l_total)
    rng_idx = randperm(l_total)[1:n_use]

    println("  Candidates scored : $n_cand")
    println("  Full candidate pool: $(d.n)")
    println("  MCMC samples      : $n_use / $l_total")
    println("  OCS calls total   : $(n_use * (1 + n_cand))")

    scores        = zeros(Float64, n_cand)
    baseline_gains = Float64[]
    prog = max(1, n_use ÷ 10)

    for (k, j) in enumerate(rng_idx)
        k % prog == 0 && println("  Progress: $k / $n_use")

        g_j = d.mcmc_matrix[j, :]

        # Baseline: unconstrained OCS on all 900 candidates
        c_base    = run_ocs(d.G, g_j, d.sire_idx, d.dam_idx, THETA)
        gain_base = dot(c_base, g_j)
        push!(baseline_gains, gain_base)

        # For each candidate: constrain c_i = 0 and rerun on all 900
        for (ci, ind) in enumerate(candidate_indices)
            c_constrained = run_ocs_exclude(d.G, g_j, d.sire_idx, d.dam_idx,
                                            THETA, ind)
            gain_constrained = dot(c_constrained, g_j)
            scores[ci] += (gain_base - gain_constrained)
        end
    end

    scores ./= n_use
    mean_base  = mean(baseline_gains)
    pct_impact = (scores ./ mean_base) .* 100.0
    standardised = length(scores) > 1 ?
        (scores .- mean(scores)) ./ std(scores) : zeros(length(scores))

    sel_map  = [d.c_map[ind]    > SELECTION_THRESHOLD for ind in candidate_indices]
    sel_cvar = [d.c_cvar[ind]   > SELECTION_THRESHOLD for ind in candidate_indices]
    sel_orc  = [d.c_oracle[ind] > SELECTION_THRESHOLD for ind in candidate_indices]

    results = DataFrame(
        individual           = candidate_indices,
        individual_id        = d.lastgen_ids[candidate_indices],
        ebv_index            = d.g_map[candidate_indices],
        tbv_index            = d.tbv_index[candidate_indices],
        map_contribution     = d.c_map[candidate_indices],
        cvar_contribution    = d.c_cvar[candidate_indices],
        oracle_contribution  = d.c_oracle[candidate_indices],
        selected_map         = sel_map,
        selected_cvar        = sel_cvar,
        selected_oracle      = sel_orc,
        robustness_score     = scores,
        percentage_impact    = pct_impact,
        standardised_score   = standardised,
        sex                  = d.sex_vec[candidate_indices]
    )

    results.selection_group = map(eachrow(results)) do r
        if r.selected_map && r.selected_cvar;    "Shared (MAP+CVaR)"
        elseif r.selected_map && !r.selected_cvar; "MAP-only (dropped)"
        elseif !r.selected_map && r.selected_cvar; "CVaR-only (recruited)"
        else;                                       "Unselected"
        end
    end

    sort!(results, :robustness_score, rev=true)

    println("\n  Score range: [$(round(minimum(scores),digits=6)), $(round(maximum(scores),digits=6))]")
    println("  All scores ≥ 0: $(all(scores .>= -1e-10))")
    println("  Impact range: [$(round(minimum(pct_impact),digits=3))%, $(round(maximum(pct_impact),digits=3))%]")
    for grp in ["Shared (MAP+CVaR)", "MAP-only (dropped)", "CVaR-only (recruited)"]
        sub = scores[[results.selection_group[i] == grp for i in 1:nrow(results)]]
        isempty(sub) && continue
        @printf("  %-28s  n=%2d  mean=%.5f  sd=%.5f\n",
                grp, length(sub), mean(sub), std(sub))
    end

    return results, baseline_gains

end
# =============================================================================

"""
    evaluate_solutions(d) -> DataFrame

Evaluate MAP-OCS and CVaR-OCS gain distributions across all MCMC iterations.
Returns long-format DataFrame with columns: iteration, solution, gain.
"""
function evaluate_solutions(d::NamedTuple)
    println("\n" * "=" ^ 70)
    println("SOLUTION EVALUATION ACROSS MCMC ITERATIONS")
    println("=" ^ 70)

    l = size(d.mcmc_matrix, 1)
    gains_map  = zeros(Float64, l)
    gains_cvar = zeros(Float64, l)

    prog = max(1, l ÷ 20)
    for j in 1:l
        j % prog == 0 && println("  Progress: $j / $l")
        g_j           = d.mcmc_matrix[j, :]
        gains_map[j]  = dot(d.c_map,  g_j)
        gains_cvar[j] = dot(d.c_cvar, g_j)
    end

    @printf("\nMAP-OCS :  mean=%.4f  sd=%.4f  CVaR95=%.4f  VaR95=%.4f\n",
            mean(gains_map), std(gains_map),
            mean(gains_map[gains_map .< quantile(gains_map, 0.05)]),
            quantile(gains_map, 0.05))
    @printf("CVaR-OCS:  mean=%.4f  sd=%.4f  CVaR95=%.4f  VaR95=%.4f\n",
            mean(gains_cvar), std(gains_cvar),
            mean(gains_cvar[gains_cvar .< quantile(gains_cvar, 0.05)]),
            quantile(gains_cvar, 0.05))

    cvar95_map  = mean(gains_map[gains_map   .< quantile(gains_map,  0.05)])
    cvar95_cvar = mean(gains_cvar[gains_cvar .< quantile(gains_cvar, 0.05)])
    @printf("CVaR95 improvement: %+.2f%%\n",
            (cvar95_cvar - cvar95_map) / abs(cvar95_map) * 100)

    return DataFrame(
        iteration = repeat(1:l, 2),
        solution  = vcat(fill("MAP-OCS", l), fill("CVaR-OCS", l)),
        gain      = vcat(gains_map, gains_cvar)
    )
end

# =============================================================================
# STATISTICAL TESTS
# =============================================================================

"""
    run_statistical_tests(eval_df) -> DataFrame

t-test, F-test, KS-test, Mann-Whitney U, Cohen's d comparing MAP vs CVaR gains.
"""
function run_statistical_tests(eval_df::DataFrame)
    println("\n" * "=" ^ 70)
    println("STATISTICAL TESTS")
    println("=" ^ 70)

    g_map  = eval_df[eval_df.solution .== "MAP-OCS",  :gain]
    g_cvar = eval_df[eval_df.solution .== "CVaR-OCS", :gain]

    tt   = UnequalVarianceTTest(g_cvar, g_map)
    ft   = VarianceFTest(g_map, g_cvar)
    ks   = ApproximateTwoSampleKSTest(g_map, g_cvar)
    mw   = MannWhitneyUTest(g_map, g_cvar)
    cohd = (mean(g_cvar) - mean(g_map)) / sqrt((var(g_map) + var(g_cvar)) / 2)

    results = DataFrame(
        test      = ["t-test (mean)", "F-test (variance)",
                     "KS-test (distribution)", "Mann-Whitney U", "Cohen's d"],
        statistic = [tt.t, ft.F, ks.δ, mw.U, cohd],
        p_value   = [pvalue(tt), pvalue(ft), pvalue(ks), pvalue(mw), NaN],
        sig_05    = [pvalue(tt)<0.05, pvalue(ft)<0.05,
                     pvalue(ks)<0.05, pvalue(mw)<0.05, abs(cohd)>0.2]
    )

    for r in eachrow(results)
        sig = r.sig_05 ? " ***" : ""
        isnan(r.p_value) ?
            println("  $(r.test): d = $(round(r.statistic, digits=4))") :
            println("  $(r.test): p = $(round(r.p_value, digits=6))$sig")
    end

    return results
end

# =============================================================================
# 4-PANEL FIGURE
# =============================================================================

"""
    create_robustness_figure(rob_df, eval_df) -> Plot

4-panel publication figure:
  (a) Robustness scores by selection group (box/violin)
  (b) EBV vs robustness score, coloured by selection group
  (c) Robustness score vs oracle membership
  (d) MCMC selection frequency vs robustness score
"""
function create_robustness_figure(rob_df::DataFrame,
                                  eval_df::DataFrame)
    println("\n" * "=" ^ 70)
    println("BUILDING 4-PANEL FIGURE")
    println("=" ^ 70)

    grp_colors = Dict(
        "Shared (MAP+CVaR)"    => C_MAP,
        "MAP-only (dropped)"   => RGB(0.85, 0.20, 0.20),
        "CVaR-only (recruited)"=> C_CVAR,
        "Unselected"           => C_UNSEL
    )
    grp_order = ["MAP-only (dropped)", "Shared (MAP+CVaR)", "CVaR-only (recruited)"]

    # ── Panel (a): Robustness score by selection group ──────────────────────
    println("  (a) Robustness by selection group...")
    pa = plot(xlabel="Selection group", ylabel="Robustness score",
              title="(a) Robustness by selection status",
              grid=false, framestyle=:box, legend=false,
              xrotation=15, bottom_margin=8Plots.mm)

    for (gi, grp) in enumerate(grp_order)
        sub = filter(r -> r.selection_group == grp, eachrow(rob_df))
        isempty(sub) && continue
        scores = [r.robustness_score for r in sub]
        col = grp_colors[grp]

        # Box: quartiles + whiskers manually
        q1, med, q3 = quantile(scores, [0.25, 0.50, 0.75])
        iqr = q3 - q1
        lo  = max(minimum(scores), q1 - 1.5*iqr)
        hi  = min(maximum(scores), q3 + 1.5*iqr)

        # jittered points
        jitter = 0.12 .* (rand(length(scores)) .- 0.5)
        scatter!(pa, fill(gi, length(scores)) .+ jitter, scores,
                 color=col, alpha=0.5, markersize=4, markerstrokewidth=0, label="")
        # box body
        plot!(pa, [gi-0.2, gi+0.2, gi+0.2, gi-0.2, gi-0.2],
              [q1, q1, q3, q3, q1], color=col, linewidth=2, label="")
        # median line
        plot!(pa, [gi-0.2, gi+0.2], [med, med], color=col, linewidth=3, label="")
        # whiskers
        plot!(pa, [gi, gi], [lo, q1], color=col, linewidth=1.5, label="")
        plot!(pa, [gi, gi], [q3, hi], color=col, linewidth=1.5, label="")
    end

    xticks!(pa, 1:length(grp_order),
            ["MAP-only\n(dropped)", "Shared\n(MAP+CVaR)", "CVaR-only\n(recruited)"])

    # ── Panel (b): EBV vs robustness, coloured by group ─────────────────────
    println("  (b) EBV vs robustness score...")
    pb = plot(xlabel="Standardised EBV index",
              ylabel="Robustness score",
              title="(b) EBV vs robustness by selection status",
              grid=false, framestyle=:box, legend=:topleft)

    for grp in ["Unselected", "Shared (MAP+CVaR)",
                "MAP-only (dropped)", "CVaR-only (recruited)"]
        sub = filter(r -> r.selection_group == grp, eachrow(rob_df))
        isempty(sub) && continue
        xs = [r.ebv_index       for r in sub]
        ys = [r.robustness_score for r in sub]
        col = grp_colors[grp]
        ms = grp == "Unselected" ? 3 : 6
        al = grp == "Unselected" ? 0.25 : 0.80
        mk = grp == "CVaR-only (recruited)" ? :diamond :
             grp == "MAP-only (dropped)"    ? :xcross  : :circle
        scatter!(pb, xs, ys, color=col, alpha=al, markersize=ms,
                 marker=mk, markerstrokewidth=0, label=grp)
    end

    # ── Panel (c): Robustness vs oracle membership ───────────────────────────
    println("  (c) Robustness vs oracle membership...")
    oracle_groups = ["Not in oracle", "In oracle"]
    pc = plot(xlabel="Oracle membership",
              ylabel="Robustness score",
              title="(c) Robustness score vs oracle membership",
              grid=false, framestyle=:box, legend=false,
              bottom_margin=5Plots.mm)

    for (gi, in_oracle) in enumerate([false, true])
        sub = filter(r -> r.selected_oracle == in_oracle, eachrow(rob_df))
        isempty(sub) && continue
        scores = [r.robustness_score for r in sub]
        col = in_oracle ? C_ORACLE : C_UNSEL

        q1, med, q3 = quantile(scores, [0.25, 0.50, 0.75])
        iqr = q3 - q1
        lo  = max(minimum(scores), q1 - 1.5*iqr)
        hi  = min(maximum(scores), q3 + 1.5*iqr)

        jitter = 0.10 .* (rand(length(scores)) .- 0.5)
        scatter!(pc, fill(gi, length(scores)) .+ jitter, scores,
                 color=col, alpha=0.5, markersize=4, markerstrokewidth=0, label="")
        plot!(pc, [gi-0.2, gi+0.2, gi+0.2, gi-0.2, gi-0.2],
              [q1, q1, q3, q3, q1], color=col, linewidth=2, label="")
        plot!(pc, [gi-0.2, gi+0.2], [med, med], color=col, linewidth=3, label="")
        plot!(pc, [gi, gi], [lo, q1], color=col, linewidth=1.5, label="")
        plot!(pc, [gi, gi], [q3, hi], color=col, linewidth=1.5, label="")

        n_sub = length(scores)
        annotate!(pc, gi, maximum(scores) + 0.002,
                  text("n=$n_sub", 8, :center, col))
    end
    xticks!(pc, 1:2, oracle_groups)

    # ── Panel (d): Gain distributions MAP vs CVaR ────────────────────────────
    println("  (d) Gain distributions MAP-OCS vs CVaR-OCS...")
    g_map  = eval_df[eval_df.solution .== "MAP-OCS",  :gain]
    g_cvar = eval_df[eval_df.solution .== "CVaR-OCS", :gain]

    pd = plot(xlabel="Genetic gain (in-sample)",
              ylabel="Density",
              title="(d) Gain distribution across MCMC scenarios",
              grid=false, framestyle=:box, legend=:topleft)

    density!(pd, g_map,  color=C_MAP,  linewidth=2.5,
             label="MAP-OCS  (CVaR₉₅=$(round(mean(g_map[g_map .< quantile(g_map,0.05)]),digits=3)))")
    density!(pd, g_cvar, color=C_CVAR, linewidth=2.5,
             label="CVaR-OCS (CVaR₉₅=$(round(mean(g_cvar[g_cvar .< quantile(g_cvar,0.05)]),digits=3)))")

    # Shade tail regions
    q05_map  = quantile(g_map,  0.05)
    q05_cvar = quantile(g_cvar, 0.05)
    vline!(pd, [q05_map],  color=C_MAP,  linestyle=:dash, linewidth=1.5, label="VaR₉₅ MAP")
    vline!(pd, [q05_cvar], color=C_CVAR, linestyle=:dash, linewidth=1.5, label="VaR₉₅ CVaR")

    # ── Assemble ─────────────────────────────────────────────────────────────
    fig = plot(pa, pb, pc, pd,
               layout=(2, 2),
               size=(1400, 1000),
               left_margin=10Plots.mm, bottom_margin=10Plots.mm,
               top_margin=5Plots.mm,  right_margin=5Plots.mm,
               dpi=300)

    return fig
end

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

"""
    run_analysis() -> Dict

Execute the complete robustness analysis pipeline.
"""
function run_analysis()
    println("=" ^ 70)
    println("QTL-MAS 2010 MCMC ROBUSTNESS ANALYSIS")
    println("MAP-OCS vs CVaR-OCS  |  Theta = $THETA  |  CVaR model: $CVAR_LABEL")
    println("=" ^ 70)

    # 1. Load all data
    d = load_data()

    # 2. Define candidates = union of MAP-OCS and CVaR-OCS selected
    sel_map  = findall(d.c_map  .> SELECTION_THRESHOLD)
    sel_cvar = findall(d.c_cvar .> SELECTION_THRESHOLD)
    candidates = sort(unique(vcat(sel_map, sel_cvar)))

    n_shared   = length(intersect(Set(sel_map), Set(sel_cvar)))
    n_map_only = length(setdiff(Set(sel_map),  Set(sel_cvar)))
    n_cvar_only = length(setdiff(Set(sel_cvar), Set(sel_map)))

    println("\nCandidate set:")
    println("  MAP-OCS selected     : $(length(sel_map))")
    println("  CVaR-OCS selected    : $(length(sel_cvar))")
    println("  Shared               : $n_shared")
    println("  MAP-only (dropped)   : $n_map_only")
    println("  CVaR-only (recruited): $n_cvar_only")
    println("  Total candidates     : $(length(candidates))")

    # 3. Calculate robustness scores
    rob_df, baseline_gains = calculate_robustness_scores(d, candidates,
                                                         n_samples=N_ROBUSTNESS_SAMPLES)

    # 4. Evaluate both solutions across all MCMC iterations
    eval_df = evaluate_solutions(d)

    # 5. Statistical tests
    stat_df = run_statistical_tests(eval_df)

    # 6. Figure
    fig = create_robustness_figure(rob_df, eval_df)

    # 7. Save
    println("\n" * "=" ^ 70)
    println("SAVING RESULTS")
    println("=" ^ 70)

    mkpath(FIGURES_DIR)

    rob_file  = joinpath(SAVE_DIR, "robustness_analysis_qtlmas.csv")
    eval_file = joinpath(SAVE_DIR, "evaluation_gains_qtlmas.csv")
    stat_file = joinpath(SAVE_DIR, "statistical_tests_qtlmas.csv")
    fig_pdf   = joinpath(FIGURES_DIR, "robustness_figure_qtlmas.pdf")
    fig_png   = joinpath(FIGURES_DIR, "robustness_figure_qtlmas.png")

    CSV.write(rob_file,  rob_df)
    CSV.write(eval_file, eval_df)
    CSV.write(stat_file, stat_df)
    savefig(fig, fig_pdf)
    savefig(fig, fig_png)

    println("  ✓ $(rob_file)")
    println("  ✓ $(eval_file)")
    println("  ✓ $(stat_file)")
    println("  ✓ $(fig_pdf)")
    println("  ✓ $(fig_png)")

    println("\n" * "=" ^ 70)
    println("ANALYSIS COMPLETE")
    println("=" ^ 70)

    return Dict(
        "data"            => d,
        "robustness"      => rob_df,
        "evaluation"      => eval_df,
        "statistics"      => stat_df,
        "candidates"      => candidates,
        "figure"          => fig,
        "baseline_gains"  => baseline_gains
    )
end

# =============================================================================
# EXECUTION
# =============================================================================

println("""
$(repeat("=", 70))
QTL-MAS ROBUSTNESS ANALYSIS — READY
$(repeat("=", 70))

Configuration:
  Theta         = $THETA
  CVaR model    = $CVAR_LABEL
  MCMC samples  = $N_ROBUSTNESS_SAMPLES (for scoring; all used for evaluation)
  Save dir      = $SAVE_DIR

Run with:
    results = run_analysis()

Outputs:
  robustness_analysis_qtlmas.csv  — per-individual scores + selection status
  evaluation_gains_qtlmas.csv     — gain per iteration for MAP and CVaR
  statistical_tests_qtlmas.csv    — hypothesis test results
  robustness_figure_qtlmas.pdf/png — 4-panel publication figure
$(repeat("=", 70))
""")
