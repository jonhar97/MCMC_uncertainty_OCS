"""
MCMC Robustness Analysis — Norway Spruce and Loblolly Pine
===========================================================

Calculates robustness scores for MAP-OCS and CVaR-OCS selected individuals,
then compares the two solutions across all MCMC scenarios.

Set SPECIES = "spruce" or "pine" at the top — everything else is automatic.

No sex constraints. No TBV oracle (real data).
Candidates = union(MAP-OCS selected, CVaR-OCS selected).

Author: Jon Ahlinder (Skogforsk) / Ahlinder & Waldmann
"""

if !@isdefined(FOREST_ROBUST_LOADED)
    global FOREST_ROBUST_LOADED = true
end

using DataFrames, CSV, Statistics, LinearAlgebra
using COSMO, JuMP, Plots, StatsPlots
using DelimitedFiles, Measures, JLD2, FileIO
using HypothesisTests, Distributions, Printf
using Random

# =============================================================================
# SPECIES CONFIGURATION
# =============================================================================

SPECIES = "pine"    # "spruce" or "pine"

if SPECIES == "spruce"

    SPECIES_LABEL   = "Norway Spruce (Picea abies)"
    N_EXPECTED      = 1218
    THETA           = 0.02
    CVAR_LABEL      = "CVaR_a0.90_mu1.50"   # adjust to best model from CVaR run

    BASE_DIR        = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\"
    SAVE_DIR        = joinpath(BASE_DIR, "Save")
    RESULTS_DIR     = joinpath(BASE_DIR, "results_JWAS_1218_G_adj_Lev17")

    GRM_FILE        = joinpath(SAVE_DIR, "Gmat_1218_spruce_PDF.txt")

    EBV_FILES       = Dict(
        "Hjd17"    => joinpath(RESULTS_DIR, "EBV_Hjd17.txt"),
        "Htv17"    => joinpath(RESULTS_DIR, "EBV_Htv17.txt"),
        "Sprant17" => joinpath(RESULTS_DIR, "EBV_Sprant17.txt")
    )

    MCMC_FILES      = Dict(
        "Hjd17"    => joinpath(RESULTS_DIR, "MCMC_samples_EBV_Hjd17.txt"),
        "Htv17"    => joinpath(RESULTS_DIR, "MCMC_samples_EBV_Htv17.txt"),
        "Sprant17" => joinpath(RESULTS_DIR, "MCMC_samples_EBV_Sprant17.txt")
    )

    TRAIT_KEYS      = ["Hjd17", "Htv17", "Sprant17"]
    TRAIT_SIGNS     = [+1.0,    +1.0,    -1.0]
    INDEX_DENOM     = 3.0

    # CVaR-OCS output directory (must match cvar_ocs_forest_trees.jl)
    CVAR_DIR        = joinpath(SAVE_DIR, "CVaR_OCS_spruce_theta0p02")

elseif SPECIES == "pine"

    SPECIES_LABEL   = "Loblolly Pine (Pinus taeda)"
    N_EXPECTED      = 926
    THETA           = 0.03
    CVAR_LABEL      = "CVaR_a0.95_mu2.00"   # adjust to best model from CVaR run

    BASE_DIR        = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData\\"
    SAVE_DIR        = joinpath(BASE_DIR, "Save")
    RESULTS_DIR     = joinpath(BASE_DIR, "results_G_926")   # adjust if different

    GRM_FILE        = joinpath(BASE_DIR, "G_926_MAF001_mis005_rrBLUP_em_JWAS.txt")  # adjust filename

    EBV_FILES       = Dict(
        "HT6"  => joinpath(RESULTS_DIR, "EBV_HT6.txt"),
        "DBH6" => joinpath(RESULTS_DIR, "EBV_DBH6.txt"),
        "WDN4" => joinpath(RESULTS_DIR, "EBV_WDN4.txt"),
        "GV6"  => joinpath(RESULTS_DIR, "EBV_GV6.txt")
    )

    MCMC_FILES      = Dict(
        "HT6"  => joinpath(RESULTS_DIR, "MCMC_samples_EBV_HT6.txt"),
        "DBH6" => joinpath(RESULTS_DIR, "MCMC_samples_EBV_DBH6.txt"),
        "WDN4" => joinpath(RESULTS_DIR, "MCMC_samples_EBV_WDN4.txt"),
        "GV6"  => joinpath(RESULTS_DIR, "MCMC_samples_EBV_GV6.txt")
    )

    TRAIT_KEYS      = ["HT6", "DBH6", "WDN4", "GV6"]
    TRAIT_SIGNS     = [+1.0,  +1.0,   +1.0,   -1.0]
    INDEX_DENOM     = 4.0

    CVAR_DIR        = joinpath(SAVE_DIR, "CVaR_OCS_pine_theta0p03")

else
    error("SPECIES must be \"spruce\" or \"pine\"")
end

# Shared parameters
SELECTION_THRESHOLD  = 1e-4
N_ROBUSTNESS_SAMPLES = 50    # MCMC iterations for robustness scoring
FIGURES_DIR          = joinpath(BASE_DIR, "Figures")
mkpath(FIGURES_DIR)

# Colour palette (consistent with CVaR visualisation scripts)
const C_MAP  = RGB(0.20, 0.40, 0.65)
const C_CVAR = RGB(0.17, 0.63, 0.17)
const C_UNSEL = RGB(0.75, 0.75, 0.75)

println("=" ^ 70)
println("MCMC Robustness Analysis — Forest Tree Breeding")
println(SPECIES_LABEL)
println("Θ=$THETA  |  CVaR model: $CVAR_LABEL")
println("=" ^ 70)

# =============================================================================
# HELPERS: INDEX CONSTRUCTION
# =============================================================================

function standardise_vec(v::Vector{Float64})
    valid = filter(!isnan, v)
    isempty(valid) && return zeros(length(v))
    μ = mean(valid); σ = std(valid)
    return σ > 0 ? (v .- μ) ./ σ : zeros(length(v))
end

function standardise_row(v::Vector{Float64})
    valid = filter(!isnan, v)
    isempty(valid) && return zeros(length(v))
    μ = mean(valid); σ = std(valid)
    return σ > 0 ? (v .- μ) ./ σ : zeros(length(v))
end

function extract_ebv_matrix(df::DataFrame)
    col_name = String(names(df)[1])
    first_col = df[:, 1]
    if occursin(r"^(ID|id|iter|Iter|iteration)"i, col_name)
        return Matrix{Float64}(df[:, 2:end])
    end
    if eltype(first_col) <: Integer && all(first_col .== 1:length(first_col))
        return Matrix{Float64}(df[:, 2:end])
    end
    return Matrix{Float64}(df)
end

# =============================================================================
# OCS FUNCTION (no sex constraints)
# =============================================================================

function run_ocs(G::Matrix{Float64}, g::Vector{Float64}, theta::Float64)
    n = length(g)
    model = Model(optimizer_with_attributes(
        COSMO.Optimizer, "max_iter" => 50000,
        "eps_abs" => 1e-4, "eps_rel" => 1e-4, "verbose" => false))
    @variable(model, c[1:n] >= 0)
    @constraint(model, sum(c) == 1.0)
    @constraint(model, 0.5 * c' * G * c <= theta)
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
# 1. LOAD DATA
# =============================================================================

function load_data()
    println("\n" * "=" ^ 70)
    println("LOADING DATA")
    println("=" ^ 70)

    # ── GRM ──────────────────────────────────────────────────────────────────
    println("\n[1] Loading GRM...")
    grm_raw = readdlm(GRM_FILE, ',', Float64, '\n', header=false)
    grm_ids = Int.(grm_raw[:, 1])
    G       = grm_raw[:, 2:end]
    n       = size(G, 1)
    @assert n == N_EXPECTED "Expected $N_EXPECTED individuals, got $n"
    println("  GRM: $n × $n  (diag mean = $(round(mean(diag(G)),digits=4)))")

    # ── MAP EBV index ─────────────────────────────────────────────────────────
    println("\n[2] Loading MAP EBVs...")
    ebv_vecs = Vector{Vector{Float64}}()
    for key in TRAIT_KEYS
        df = CSV.read(EBV_FILES[key], DataFrame, delim=',', missingstring="NA")
        push!(ebv_vecs, Float64.(df[:, 2]))
    end
    g_map = zeros(n)
    for (v, s) in zip(ebv_vecs, TRAIT_SIGNS)
        g_map .+= s .* standardise_vec(v)
    end
    g_map ./= INDEX_DENOM
    replace!(g_map, NaN => 0.0)
    println("  MAP index range: [$(round(minimum(g_map),digits=3)), $(round(maximum(g_map),digits=3))]")

    # ── MCMC scenario matrix ──────────────────────────────────────────────────
    println("\n[3] Loading MCMC chains...")
    cache_file = "tmp" # joinpath(CVAR_DIR, "mcmc_index_cache.jld2")

    local U, l_scenarios
    if isfile(cache_file)
        println("  Loading from CVaR cache: $cache_file")
        @load cache_file U l_scenarios
        println("  ✓ $l_scenarios scenarios × $n individuals")
    else
        println("  Cache not found — rebuilding from chains...")
        ebv_mats = Vector{Matrix{Float64}}()
        for key in TRAIT_KEYS
            println("  Loading $key...")
            df  = CSV.read(MCMC_FILES[key], DataFrame, delim=',',
                           missingstring="NA", header=true)
            mat = extract_ebv_matrix(df)
            @assert size(mat, 2) == n "Chain $key: $(size(mat,2)) cols ≠ $n"
            push!(ebv_mats, mat)
        end
        l_scenarios = size(ebv_mats[1], 1)
        U = zeros(l_scenarios, n)
        for j in 1:l_scenarios
            j % 100 == 0 && println("  Building cache: $j / $l_scenarios")
            row = zeros(n)
            for (M, s) in zip(ebv_mats, TRAIT_SIGNS)
                row .+= s .* standardise_row(M[j, :])
            end
            U[j, :] = row ./ INDEX_DENOM
        end
        @save cache_file U l_scenarios
        println("  ✓ Cache saved: $cache_file")
    end

    # ── CVaR-OCS contribution vector ──────────────────────────────────────────
    println("\n[4] Loading CVaR-OCS contributions ($CVAR_LABEL)...")
    sol_file = joinpath(CVAR_DIR, "cvar_ocs_solutions_$(SPECIES).csv")
    sol_df   = CSV.read(sol_file, DataFrame)
    id_cols  = names(sol_df)[4:end]
    sol_id_lookup = Dict(parse(Int, replace(col, "ID_"=>"")) => col for col in id_cols)

    function load_contrib(lbl::String)
        row = sol_df[String.(sol_df.label) .== lbl, :]
        isempty(row) && error("Label '$lbl' not found in $sol_file\nAvailable: $(unique(String.(sol_df.label)))")
        return [haskey(sol_id_lookup, id) ? Float64(row[1, sol_id_lookup[id]]) : 0.0
                for id in grm_ids]
    end

    c_map_loaded = load_contrib("MAP-OCS")
    c_cvar       = load_contrib(CVAR_LABEL)

    @printf("  MAP-OCS selected  : %d\n", sum(c_map_loaded .> SELECTION_THRESHOLD))
    @printf("  CVaR-OCS selected : %d\n", sum(c_cvar       .> SELECTION_THRESHOLD))

    return (G=G, n=n, grm_ids=grm_ids, g_map=g_map,
            U=U, l=l_scenarios,
            c_map=c_map_loaded, c_cvar=c_cvar)
end

# =============================================================================
# 2. ROBUSTNESS SCORES
# =============================================================================

"""
    run_ocs_exclude(G, g, theta, exclude_ind) -> Vector{Float64}

OCS with hard constraint c[exclude_ind] = 0. No sex constraints (forest trees).
All other candidates remain available — solver redistributes optimally.
"""
function run_ocs_exclude(G::Matrix{Float64}, g::Vector{Float64},
                         theta::Float64, exclude_ind::Int)
    n = length(g)
    model = Model(optimizer_with_attributes(
        COSMO.Optimizer, "max_iter" => 50000,
        "eps_abs" => 1e-4, "eps_rel" => 1e-4, "verbose" => false))
    @variable(model, c[1:n] >= 0)
    @constraint(model, c[exclude_ind] == 0.0)
    @constraint(model, sum(c) == 1.0)
    @constraint(model, 0.5 * c' * G * c <= theta)
    @objective(model, Max, dot(g, c))
    optimize!(model)
    if termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        return value.(c)
    else
        @warn "run_ocs_exclude did not converge for ind=$exclude_ind"
        return zeros(n)
    end
end

function calculate_robustness_scores(d::NamedTuple, candidates::Vector{Int};
                                     n_samples::Int=N_ROBUSTNESS_SAMPLES)
    println("\n" * "=" ^ 70)
    println("ROBUSTNESS SCORE CALCULATION (constrained exclusion)")
    println("=" ^ 70)
    println("  Method: hard constraint c_i=0, OCS re-run on all $N_EXPECTED candidates")
    println("  Scores ≥ 0: gain loss from excluding individual i")

    n_cand  = length(candidates)
    n_use   = min(n_samples, d.l)
    rng     = randperm(d.l)[1:n_use]
    println("  Candidates scored  : $n_cand")
    println("  Full candidate pool: $(d.n)")
    println("  MCMC samples       : $n_use / $(d.l)")
    println("  OCS calls total    : $(n_use * (1 + n_cand))")

    scores        = zeros(Float64, n_cand)
    baseline_gains = Float64[]
    prog = max(1, n_use ÷ 10)

    for (k, j) in enumerate(rng)
        k % prog == 0 && println("  Progress: $k / $n_use")
        g_j = d.U[j, :]

        # Baseline: unconstrained OCS on all candidates
        c_base    = run_ocs(d.G, g_j, THETA)
        gain_base = dot(c_base, g_j)
        push!(baseline_gains, gain_base)

        # For each candidate: hard constraint c_i=0, rerun on all candidates
        for (ci, ind) in enumerate(candidates)
            c_ex         = run_ocs_exclude(d.G, g_j, THETA, ind)
            gain_ex      = dot(c_ex, g_j)
            scores[ci]  += (gain_base - gain_ex)
        end
    end

    scores ./= n_use
    mean_base    = mean(baseline_gains)
    pct_impact   = (scores ./ mean_base) .* 100.0
    standardised = length(scores) > 1 ?
        (scores .- mean(scores)) ./ std(scores) : zeros(length(scores))

    sel_map_v  = [d.c_map[i]  > SELECTION_THRESHOLD for i in candidates]
    sel_cvar_v = [d.c_cvar[i] > SELECTION_THRESHOLD for i in candidates]

    results = DataFrame(
        individual         = candidates,
        individual_id      = d.grm_ids[candidates],
        ebv_index          = d.g_map[candidates],
        map_contribution   = d.c_map[candidates],
        cvar_contribution  = d.c_cvar[candidates],
        selected_map       = sel_map_v,
        selected_cvar      = sel_cvar_v,
        robustness_score   = scores,
        percentage_impact  = pct_impact,
        standardised_score = standardised
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
    for grp in ["Shared (MAP+CVaR)", "MAP-only (dropped)", "CVaR-only (recruited)"]
        sub = scores[[results.selection_group[i] == grp for i in 1:nrow(results)]]
        isempty(sub) && continue
        @printf("  %-28s  n=%3d  mean=%.5f  sd=%.5f\n",
                grp, length(sub), mean(sub), std(sub))
    end
    return results, baseline_gains
end
# =============================================================================

function evaluate_solutions(d::NamedTuple)
    println("\n" * "=" ^ 70)
    println("SOLUTION EVALUATION ACROSS ALL MCMC ITERATIONS")
    println("=" ^ 70)

    gains_map  = [dot(d.c_map,  d.U[j, :]) for j in 1:d.l]
    gains_cvar = [dot(d.c_cvar, d.U[j, :]) for j in 1:d.l]

    for (lbl, gains) in [("MAP-OCS", gains_map), ("CVaR-OCS", gains_cvar)]
        cvar95 = mean(gains[gains .< quantile(gains, 0.05)])
        @printf("  %-10s  mean=%.4f  sd=%.4f  VaR95=%.4f  CVaR95=%.4f\n",
                lbl, mean(gains), std(gains), quantile(gains, 0.05), cvar95)
    end

    cvar95_map  = mean(gains_map[gains_map   .< quantile(gains_map,  0.05)])
    cvar95_cvar = mean(gains_cvar[gains_cvar .< quantile(gains_cvar, 0.05)])
    @printf("  CVaR95 improvement: %+.2f%%\n",
            (cvar95_cvar - cvar95_map) / abs(cvar95_map) * 100)

    return DataFrame(
        iteration = repeat(1:d.l, 2),
        solution  = vcat(fill("MAP-OCS", d.l), fill("CVaR-OCS", d.l)),
        gain      = vcat(gains_map, gains_cvar)
    )
end

# =============================================================================
# 4. STATISTICAL TESTS
# =============================================================================

function run_statistical_tests(eval_df::DataFrame)
    g_map  = eval_df[eval_df.solution .== "MAP-OCS",  :gain]
    g_cvar = eval_df[eval_df.solution .== "CVaR-OCS", :gain]
    tt   = UnequalVarianceTTest(g_cvar, g_map)
    ft   = VarianceFTest(g_map, g_cvar)
    ks   = ApproximateTwoSampleKSTest(g_map, g_cvar)
    mw   = MannWhitneyUTest(g_map, g_cvar)
    cohd = (mean(g_cvar)-mean(g_map)) / sqrt((var(g_map)+var(g_cvar))/2)
    results = DataFrame(
        test      = ["t-test","F-test","KS-test","Mann-Whitney U","Cohen's d"],
        statistic = [tt.t, ft.F, ks.δ, mw.U, cohd],
        p_value   = [pvalue(tt), pvalue(ft), pvalue(ks), pvalue(mw), NaN]
    )
    println("\n" * "=" ^ 70)
    println("STATISTICAL TESTS")
    println("=" ^ 70)
    for r in eachrow(results)
        isnan(r.p_value) ?
            @printf("  %-20s  d = %.4f\n", r.test, r.statistic) :
            @printf("  %-20s  p = %.6f%s\n", r.test, r.p_value,
                    r.p_value < 0.05 ? " ***" : "")
    end
    return results
end

# =============================================================================
# 5. 4-PANEL FIGURE
# =============================================================================

function create_figure(rob_df::DataFrame, eval_df::DataFrame)
    println("\n" * "=" ^ 70)
    println("BUILDING 4-PANEL FIGURE")
    println("=" ^ 70)

    grp_colors = Dict(
        "Shared (MAP+CVaR)"     => C_MAP,
        "MAP-only (dropped)"    => RGB(0.85, 0.20, 0.20),
        "CVaR-only (recruited)" => C_CVAR,
        "Unselected"            => C_UNSEL
    )
    grp_order = ["MAP-only (dropped)", "Shared (MAP+CVaR)", "CVaR-only (recruited)"]

    # Helper: manual box with jitter
    function add_box!(p, gi, scores, col)
        q1, med, q3 = quantile(scores, [0.25, 0.50, 0.75])
        iqr = q3 - q1
        lo  = max(minimum(scores), q1 - 1.5*iqr)
        hi  = min(maximum(scores), q3 + 1.5*iqr)
        jit = 0.12 .* (rand(length(scores)) .- 0.5)
        scatter!(p, fill(gi,length(scores)).+jit, scores,
                 color=col, alpha=0.45, ms=4, markerstrokewidth=0, label="")
        plot!(p, [gi-.2,gi+.2,gi+.2,gi-.2,gi-.2], [q1,q1,q3,q3,q1],
              color=col, lw=2, label="")
        plot!(p, [gi-.2,gi+.2], [med,med], color=col, lw=3, label="")
        plot!(p, [gi,gi],[lo,q1], color=col, lw=1.5, label="")
        plot!(p, [gi,gi],[q3,hi], color=col, lw=1.5, label="")
    end

    # ── Panel (a): Robustness by selection group ────────────────────────────
    pa = plot(title="(a) Robustness by selection status",
              ylabel="Robustness score", grid=false, framestyle=:box,
              legend=false, xrotation=12, bottom_margin=8Plots.mm)
    for (gi, grp) in enumerate(grp_order)
        sub = rob_df[rob_df.selection_group .== grp, :robustness_score]
        isempty(sub) && continue
        add_box!(pa, gi, sub, grp_colors[grp])
        annotate!(pa, gi, maximum(sub)*1.02, text("n=$(length(sub))", 8, :center))
    end
    xticks!(pa, 1:length(grp_order),
            ["MAP-only\n(dropped)", "Shared\n(MAP+CVaR)", "CVaR-only\n(recruited)"])

    # ── Panel (b): EBV vs robustness ────────────────────────────────────────
    pb = plot(title="(b) EBV vs robustness score",
              xlabel="Standardised EBV index", ylabel="Robustness score",
              grid=false, framestyle=:box, legend=:topleft)
    for grp in ["Unselected","Shared (MAP+CVaR)","MAP-only (dropped)","CVaR-only (recruited)"]
        sub = rob_df[rob_df.selection_group .== grp, :]
        isempty(sub) && continue
        col = grp_colors[grp]
        ms  = grp == "Unselected" ? 3 : 6
        al  = grp == "Unselected" ? 0.20 : 0.80
        mk  = grp == "CVaR-only (recruited)" ? :diamond :
              grp == "MAP-only (dropped)"    ? :xcross  : :circle
        scatter!(pb, sub.ebv_index, sub.robustness_score,
                 color=col, alpha=al, ms=ms, marker=mk,
                 markerstrokewidth=0, label=grp)
    end

    # ── Panel (c): Gain distributions ───────────────────────────────────────
    g_map_v  = eval_df[eval_df.solution .== "MAP-OCS",  :gain]
    g_cvar_v = eval_df[eval_df.solution .== "CVaR-OCS", :gain]
    cvar95_m = mean(g_map_v[g_map_v   .< quantile(g_map_v,  0.05)])
    cvar95_c = mean(g_cvar_v[g_cvar_v .< quantile(g_cvar_v, 0.05)])

    pc = plot(title="(c) Gain distribution across MCMC scenarios",
              xlabel="Genetic gain (in-sample)", ylabel="Density",
              grid=false, framestyle=:box, legend=:topleft)
    density!(pc, g_map_v,  color=C_MAP,  lw=2.5,
             label="MAP-OCS (CVaR₉₅=$(round(cvar95_m,digits=3)))")
    density!(pc, g_cvar_v, color=C_CVAR, lw=2.5,
             label="CVaR-OCS (CVaR₉₅=$(round(cvar95_c,digits=3)))")
    vline!(pc, [quantile(g_map_v,  0.05)], color=C_MAP,  ls=:dash, lw=1.5, label="VaR₉₅ MAP")
    vline!(pc, [quantile(g_cvar_v, 0.05)], color=C_CVAR, ls=:dash, lw=1.5, label="VaR₉₅ CVaR")

    # ── Panel (d): Robustness vs contribution magnitude ─────────────────────
    pd = plot(title="(d) Robustness score vs MAP contribution",
              xlabel="MAP contribution (c)", ylabel="Robustness score",
              grid=false, framestyle=:box, legend=:topleft)
    for grp in ["Shared (MAP+CVaR)","MAP-only (dropped)","CVaR-only (recruited)"]
        sub = rob_df[rob_df.selection_group .== grp, :]
        isempty(sub) && continue
        col = grp_colors[grp]
        mk  = grp == "CVaR-only (recruited)" ? :diamond :
              grp == "MAP-only (dropped)"    ? :xcross  : :circle
        scatter!(pd, sub.map_contribution, sub.robustness_score,
                 color=col, alpha=0.80, ms=6, marker=mk,
                 markerstrokewidth=0, label=grp)
    end

    fig = plot(pa, pb, pc, pd,
               layout=(2,2), size=(1400, 1000), dpi=300,
               left_margin=10Plots.mm, bottom_margin=10Plots.mm,
               top_margin=5Plots.mm,  right_margin=5Plots.mm,
               plot_title="$(SPECIES_LABEL)  |  Θ=$(THETA)  |  $(CVAR_LABEL)")
    return fig
end

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

function run_analysis()
    println("=" ^ 70)
    println("FOREST TREE ROBUSTNESS ANALYSIS — $SPECIES_LABEL")
    println("=" ^ 70)

    # 1. Load
    d = load_data()

    # 2. Candidates = union of both selected sets
    sel_map  = findall(d.c_map  .> SELECTION_THRESHOLD)
    sel_cvar = findall(d.c_cvar .> SELECTION_THRESHOLD)
    candidates = sort(unique(vcat(sel_map, sel_cvar)))

    n_shared    = length(intersect(Set(sel_map),  Set(sel_cvar)))
    n_map_only  = length(setdiff(Set(sel_map),   Set(sel_cvar)))
    n_cvar_only = length(setdiff(Set(sel_cvar),  Set(sel_map)))
    println("\nCandidate breakdown:")
    println("  MAP-OCS selected     : $(length(sel_map))")
    println("  CVaR-OCS selected    : $(length(sel_cvar))")
    println("  Shared               : $n_shared")
    println("  MAP-only (dropped)   : $n_map_only")
    println("  CVaR-only (recruited): $n_cvar_only")
    println("  Total candidates     : $(length(candidates))")

    # 3. Robustness scores
    rob_df, baseline_gains = calculate_robustness_scores(d, candidates,
                                                         n_samples=N_ROBUSTNESS_SAMPLES)

    # 4. Evaluate both solutions
    eval_df = evaluate_solutions(d)

    # 5. Statistical tests
    stat_df = run_statistical_tests(eval_df)

    # 6. Figure
    fig = create_figure(rob_df, eval_df)

    # 7. Save
    println("\n" * "=" ^ 70)
    println("SAVING RESULTS")
    println("=" ^ 70)

    rob_file  = joinpath(CVAR_DIR, "robustness_analysis_$(SPECIES).csv")
    eval_file = joinpath(CVAR_DIR, "evaluation_gains_$(SPECIES).csv")
    stat_file = joinpath(CVAR_DIR, "statistical_tests_$(SPECIES).csv")
    fig_pdf   = joinpath(FIGURES_DIR, "robustness_figure_$(SPECIES).pdf")
    fig_png   = joinpath(FIGURES_DIR, "robustness_figure_$(SPECIES).png")

    CSV.write(rob_file,  rob_df)
    CSV.write(eval_file, eval_df)
    CSV.write(stat_file, stat_df)
    savefig(fig, fig_pdf)
    savefig(fig, fig_png)

    for f in [rob_file, eval_file, stat_file, fig_pdf, fig_png]
        println("  ✓ $f")
    end

    println("\n" * "=" ^ 70)
    println("ANALYSIS COMPLETE — $SPECIES_LABEL")
    println("=" ^ 70)

    return Dict("data"=>d, "robustness"=>rob_df,
                "evaluation"=>eval_df, "statistics"=>stat_df,
                "candidates"=>candidates, "figure"=>fig,
                "baseline_gains"=>baseline_gains)
end

# =============================================================================
# EXECUTION
# =============================================================================

println("""
$(repeat("=", 70))
FOREST TREE ROBUSTNESS ANALYSIS — READY
$(repeat("=", 70))

Species  : $SPECIES_LABEL
Theta    : $THETA
CVaR mdl : $CVAR_LABEL
Samples  : $N_ROBUSTNESS_SAMPLES (for scoring; all used for evaluation)

IMPORTANT: Run cvar_ocs_forest_trees.jl first for this species
           so that cvar_ocs_solutions_$(SPECIES).csv exists.
           Update CVAR_LABEL to the best model from that run.

Run with:
    results = run_analysis()
$(repeat("=", 70))
""")
