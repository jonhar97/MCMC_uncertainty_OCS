"""
OCS Computational Benchmarking
================================
Compares wall-clock time for:
  1. MAP-OCS (reference)                         -- timed live, 10 reps
  2. CVaR-OCS single (alpha=0.95, mu=1.0)        -- timed live, 10 reps
  3. CVaR-OCS sweep (27 alpha x mu combos)       -- extrapolated from (2)
  4. Robustness score (n_selected x mcmc_subset) -- extrapolated from (1)

Datasets:
  Spruce_1218  n=1218  Theta=0.02  l=1000 MCMC scenarios
  Spruce_5525  n=5525  Theta=0.02  l=1000 MCMC scenarios
  QTLMAS_900   n=900   Theta=0.05  l=1000 MCMC scenarios

Outputs:
  benchmark_results.csv  -- per-run timings
  benchmark_summary.csv  -- mean +/- SD, overhead factors

Author: Jon Ahlinder (Skogforsk) / Ahlinder & Waldmann
"""

using CSV, DataFrames, Statistics, LinearAlgebra
using JuMP, COSMO
using JLD2, FileIO
using Printf
using Random
using DelimitedFiles: readdlm

# =============================================================================
# CONFIGURATION
# =============================================================================

N_REPS        = 10
ALPHA_VALUES  = [0.90, 0.95, 0.99]
MU_VALUES     = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 5.0, 10.0]
L_SCENARIOS   = 1_000
SELECTION_THR = 1e-4

COSMO_MAX_ITER = 30_000
COSMO_EPS      = 1e-5

OUT_DIR = @__DIR__

# =============================================================================
# CACHED RESULTS: Spruce_1218 (from previous benchmark run, 2026-05-27)
# These are injected directly so we do not re-run those solves.
# =============================================================================
CACHED_RESULTS = [
    (dataset="Spruce_1218", method="MAP-OCS",           alpha=NaN,  mu=NaN,
     mean_s=3.863,   sd_s=0.883,  min_s=2.584,  max_s=4.755,
     n=1218, l=1000, theta=0.02,  note="timed, 10 reps (cached)"),
    (dataset="Spruce_1218", method="CVaR-OCS_single",   alpha=0.95, mu=1.0,
     mean_s=84.023,  sd_s=4.436,  min_s=74.238, max_s=89.592,
     n=1218, l=1000, theta=0.02,  note="timed, 10 reps (cached)"),
    (dataset="Spruce_5525", method="MAP-OCS",           alpha=NaN,  mu=NaN,
     mean_s=161.553, sd_s=23.027, min_s=126.720, max_s=187.106,
     n=5525, l=1000, theta=0.02,  note="timed, 10 reps (cached)"),
    (dataset="Spruce_5525", method="CVaR-OCS_single",   alpha=0.95, mu=1.0,
     mean_s=2724.182, sd_s=41.439, min_s=2684.187, max_s=2826.405,
     n=5525, l=1000, theta=0.02,  note="timed, 10 reps (cached)"),
    # sweep and robustness are extrapolated in the summary step below
]

# Spruce_1218 derived extrapolations (n_sel=88 from cached MAP-OCS solution)
const SPRUCE1218_MAP_MEAN   = 3.863
const SPRUCE1218_CVAR_MEAN  = 84.023
const SPRUCE1218_NSEL       = 88
const SPRUCE1218_N          = 1218
const SPRUCE1218_L          = 1000
const SPRUCE1218_THETA      = 0.02
const SPRUCE1218_NMCMC      = 200

# Spruce_5525 derived extrapolations (n_sel=41 from cached MAP-OCS solution)
const SPRUCE5525_MAP_MEAN   = 161.553
const SPRUCE5525_CVAR_MEAN  = 2724.182
const SPRUCE5525_NSEL       = 41
const SPRUCE5525_N          = 5525
const SPRUCE5525_L          = 1000
const SPRUCE5525_THETA      = 0.02
const SPRUCE5525_NMCMC      = 200

DATASETS = [
    (
        label             = "QTLMAS_3226",
        n_expected        = 3226,   # full GRM covers all generations
        theta             = 0.05,
        grm_file          = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\QTLMAS\\Save\\GRM_QTLMAS_jwas.txt",
        ebv_file          = "",
        mcmc_cache        = "",
        n_mcmc_robustness = 200,
    ),
]

# =============================================================================
# HELPERS
# =============================================================================

function make_optimizer()
    optimizer_with_attributes(
        COSMO.Optimizer,
        "max_iter" => COSMO_MAX_ITER,
        "eps_abs"  => COSMO_EPS,
        "eps_rel"  => COSMO_EPS,
        "verbose"  => false
    )
end

function load_grm(path::String; jitter::Float64=1e-6)
    raw = readdlm(path, ',', Float64, '\n', header=false)
    ids = Int.(raw[:, 1])
    G   = raw[:, 2:end]
    n   = size(G, 1)
    @assert size(G, 2) == n "GRM is not square"
    # Ensure symmetry and positive definiteness for COSMO.
    # Real GRMs can have tiny negative eigenvalues from floating-point
    # accumulation; a small diagonal jitter fixes this without materially
    # changing the coancestry constraint.
    G   = (G .+ G') ./ 2
    G  += jitter .* I(n)
    return G, ids, n
end

function synthetic_grm(n::Int; rng=Random.default_rng())
    Z = randn(rng, n, n) ./ sqrt(n)
    G = Z * Z'
    G ./= mean(diag(G))
    return (G .+ G') ./ 2
end

function synthetic_mcmc_scenarios(g::Vector{Float64}, l::Int;
                                  noise_sd::Float64=0.3,
                                  rng=Random.default_rng())
    n = length(g)
    return repeat(g', l, 1) .+ noise_sd .* randn(rng, l, n)
end

function run_map_ocs_timed(G::Matrix{Float64}, g::Vector{Float64}, theta::Float64)
    n  = length(g)
    t0 = time()
    model = Model(make_optimizer())
    @variable(model, c[1:n] >= 0)
    @constraint(model, sum(c) == 1.0)
    @constraint(model, 0.5 * c' * G * c <= theta)
    @objective(model, Max, dot(g, c))
    optimize!(model)
    elapsed = time() - t0
    status  = termination_status(model)
    c_val   = status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) ? value.(c) : zeros(n)
    return c_val, elapsed, status
end

function run_cvar_ocs_timed(G::Matrix{Float64}, U::Matrix{Float64},
                             theta::Float64, mu::Float64, alpha::Float64)
    l, n   = size(U)
    inv_al = 1.0 / ((1.0 - alpha) * l)
    g_bar  = vec(mean(U, dims=1))
    t0     = time()
    model  = Model(make_optimizer())
    @variable(model, c[1:n] >= 0)
    @variable(model, eta)
    @variable(model, z[1:l] >= 0)
    @constraint(model, sum(c) == 1.0)
    @constraint(model, 0.5 * c' * G * c <= theta)
    for j in 1:l
        @constraint(model, eta - dot(U[j, :], c) - z[j] <= 0)
    end
    @objective(model, Max, dot(g_bar, c) + mu * (eta - inv_al * sum(z)))
    optimize!(model)
    elapsed = time() - t0
    status  = termination_status(model)
    c_val   = status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) ? value.(c) : zeros(n)
    return c_val, elapsed, status
end

# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

function benchmark_dataset(label::String, G::Matrix{Float64},
                            g::Vector{Float64}, U::Matrix{Float64},
                            theta::Float64, n_mcmc_robustness::Int)
    n, l = size(G, 1), size(U, 1)
    rows = []

    println("\n  Dataset : $label  (n=$n, l=$l, Theta=$theta)")
    println("  Running $N_REPS repetitions per benchmark...")
    println("  (warm-up calls first to avoid JIT timing inflation)")

    # -- warm-up: one untimed solve of each type to trigger JIT compilation ---
    alpha_single, mu_single = 0.95, 1.0
    print("    [warm-up MAP-OCS] ... ")
    c_map, _, _ = run_map_ocs_timed(G, g, theta)   # result kept for n_selected
    println("done")
    print("    [warm-up CVaR-OCS] ... ")
    run_cvar_ocs_timed(G, U, theta, mu_single, alpha_single)
    println("done")

    # -- 1. MAP-OCS -- timed live ---------------------------------------------
    print("    [MAP-OCS]  rep: ")
    map_times = Float64[]
    for rep in 1:N_REPS
        print("$rep ")
        _, t, _ = run_map_ocs_timed(G, g, theta)
        push!(map_times, t)
    end
    println()
    push!(rows, (dataset=label, method="MAP-OCS", alpha=NaN, mu=NaN,
                 mean_s=mean(map_times), sd_s=std(map_times),
                 min_s=minimum(map_times), max_s=maximum(map_times),
                 n=n, l=l, theta=theta, note="timed"))
    @printf("      mean=%.3f s  sd=%.3f s  [%.3f, %.3f]\n",
            mean(map_times), std(map_times), minimum(map_times), maximum(map_times))

    # -- 2. CVaR-OCS single solve -- timed live --------------------------------
    print("    [CVaR-OCS single alpha=$(alpha_single) mu=$(mu_single)]  rep: ")
    cvar_single_times = Float64[]
    for rep in 1:N_REPS
        print("$rep ")
        _, t, _ = run_cvar_ocs_timed(G, U, theta, mu_single, alpha_single)
        push!(cvar_single_times, t)
    end
    println()
    push!(rows, (dataset=label, method="CVaR-OCS_single", alpha=alpha_single,
                 mu=mu_single,
                 mean_s=mean(cvar_single_times), sd_s=std(cvar_single_times),
                 min_s=minimum(cvar_single_times), max_s=maximum(cvar_single_times),
                 n=n, l=l, theta=theta, note="timed"))
    @printf("      mean=%.3f s  sd=%.3f s  [%.3f, %.3f]\n",
            mean(cvar_single_times), std(cvar_single_times),
            minimum(cvar_single_times), maximum(cvar_single_times))
    @printf("      overhead vs MAP-OCS: x%.2f\n",
            mean(cvar_single_times) / mean(map_times))

    # -- 3. CVaR-OCS sweep -- extrapolated from single solve ------------------
    # The sweep is n_combos independent CVaR-OCS calls (one per alpha/mu pair).
    # Cost = n_combos x mean(CVaR_single). No need to run all 27 x N_REPS solves.
    n_combos      = length(ALPHA_VALUES) * length(MU_VALUES)
    est_sweep_s   = mean(cvar_single_times) * n_combos
    est_sweep_min = est_sweep_s / 60.0
    println("    [CVaR-OCS sweep $n_combos combos -- extrapolated, not timed live]")
    @printf("      %d combos x CVaR_single mean (%.2f s) = %.1f s  (%.1f min)\n",
            n_combos, mean(cvar_single_times), est_sweep_s, est_sweep_min)
    @printf("      overhead vs MAP-OCS: x%.2f\n",
            est_sweep_s / mean(map_times))
    push!(rows, (dataset=label, method="CVaR-OCS_sweep_estimated", alpha=NaN, mu=NaN,
                 mean_s=est_sweep_s, sd_s=NaN,
                 min_s=NaN, max_s=NaN,
                 n=n, l=l, theta=theta, note="extrapolated: $n_combos x CVaR_single"))

    # -- 4. Robustness score -- extrapolated from MAP-OCS timing ---------------
    # Cost = n_selected x n_mcmc_subset MAP-OCS solves.
    # E.g. Spruce_1218: ~150 selected x 200 MCMC draws = ~30,000 solves.
    n_selected    = sum(c_map .> SELECTION_THR)
    n_ocs_calls   = n_selected * n_mcmc_robustness
    est_robust_s  = mean(map_times) * n_ocs_calls
    est_robust_min = est_robust_s / 60.0
    est_robust_par = est_robust_s / 20.0 / 60.0   # indicative: 20 cores
    println("    [Robustness score -- extrapolated, not timed live]")
    @printf("      n_selected=%d x mcmc_subset=%d = %d MAP-OCS calls\n",
            n_selected, n_mcmc_robustness, n_ocs_calls)
    @printf("      MAP-OCS mean (%.3f s) x %d = %.1f s  (%.1f min) sequential\n",
            mean(map_times), n_ocs_calls, est_robust_s, est_robust_min)
    @printf("      ~%.1f min on 20 cores (embarrassingly parallel over candidates)\n",
            est_robust_par)
    @printf("      overhead vs MAP-OCS: x%d\n", n_ocs_calls)
    push!(rows, (dataset=label, method="RobustnessScore_estimated", alpha=NaN, mu=NaN,
                 mean_s=est_robust_s, sd_s=NaN,
                 min_s=NaN, max_s=NaN,
                 n=n, l=l, theta=theta,
                 note="extrapolated: $(n_selected) x $(n_mcmc_robustness) MAP-OCS"))

    return DataFrame(rows)
end

# =============================================================================
# MAIN
# =============================================================================

println("=" ^ 70)
println("OCS COMPUTATIONAL BENCHMARK")
@printf("  N_REPS=%d  l=%d  alpha=%s  mu=%s\n",
        N_REPS, L_SCENARIOS,
        join(ALPHA_VALUES, ","),
        join(MU_VALUES, ","))
println("=" ^ 70)

all_results = DataFrame[]
rng = MersenneTwister(42)

# ── Inject cached Spruce_1218 results ────────────────────────────────────────
println("\n" * "-" ^ 70)
println("DATASET: Spruce_1218  (cached -- not re-run)")
println("-" ^ 70)

n_combos_cached  = length(ALPHA_VALUES) * length(MU_VALUES)
est_sweep_1218   = SPRUCE1218_CVAR_MEAN * n_combos_cached
est_robust_1218  = SPRUCE1218_MAP_MEAN  * SPRUCE1218_NSEL * SPRUCE1218_NMCMC

# Spruce_1218: timed rows + extrapolated rows
cached_rows_1218 = vcat(
    DataFrame(filter(r -> r.dataset == "Spruce_1218", CACHED_RESULTS)),
    DataFrame([
        (dataset="Spruce_1218", method="CVaR-OCS_sweep_estimated", alpha=NaN, mu=NaN,
         mean_s=est_sweep_1218,  sd_s=NaN, min_s=NaN, max_s=NaN,
         n=SPRUCE1218_N, l=SPRUCE1218_L, theta=SPRUCE1218_THETA,
         note="extrapolated: $(n_combos_cached) x CVaR_single (cached)"),
        (dataset="Spruce_1218", method="RobustnessScore_estimated", alpha=NaN, mu=NaN,
         mean_s=est_robust_1218, sd_s=NaN, min_s=NaN, max_s=NaN,
         n=SPRUCE1218_N, l=SPRUCE1218_L, theta=SPRUCE1218_THETA,
         note="extrapolated: $(SPRUCE1218_NSEL) x $(SPRUCE1218_NMCMC) MAP-OCS (cached)"),
    ])
)
push!(all_results, cached_rows_1218)

@printf("  MAP-OCS:        mean=%.3f s  sd=%.3f s\n", SPRUCE1218_MAP_MEAN,  0.883)
@printf("  CVaR-OCS:       mean=%.3f s  sd=%.3f s\n", SPRUCE1218_CVAR_MEAN, 4.436)
@printf("  CVaR sweep:     %.1f s  (%.1f min) [extrapolated]\n",
        est_sweep_1218, est_sweep_1218/60)
@printf("  Robustness:     %.1f s  (%.1f min) sequential  [extrapolated]\n",
        est_robust_1218, est_robust_1218/60)

# ── Inject cached Spruce_5525 results ────────────────────────────────────────
println("\n" * "-" ^ 70)
println("DATASET: Spruce_5525  (cached -- not re-run)")
println("-" ^ 70)

est_sweep_5525   = SPRUCE5525_CVAR_MEAN * n_combos_cached
est_robust_5525  = SPRUCE5525_MAP_MEAN  * SPRUCE5525_NSEL * SPRUCE5525_NMCMC

# Spruce_5525: timed rows + extrapolated rows
cached_rows_5525 = vcat(
    DataFrame(filter(r -> r.dataset == "Spruce_5525", CACHED_RESULTS)),
    DataFrame([
    (dataset="Spruce_5525", method="CVaR-OCS_sweep_estimated", alpha=NaN, mu=NaN,
     mean_s=est_sweep_5525,  sd_s=NaN, min_s=NaN, max_s=NaN,
     n=SPRUCE5525_N, l=SPRUCE5525_L, theta=SPRUCE5525_THETA,
     note="extrapolated: $(n_combos_cached) x CVaR_single (cached)"),
    (dataset="Spruce_5525", method="RobustnessScore_estimated", alpha=NaN, mu=NaN,
     mean_s=est_robust_5525, sd_s=NaN, min_s=NaN, max_s=NaN,
     n=SPRUCE5525_N, l=SPRUCE5525_L, theta=SPRUCE5525_THETA,
     note="extrapolated: $(SPRUCE5525_NSEL) x $(SPRUCE5525_NMCMC) MAP-OCS (cached)"),
    ])
)
push!(all_results, cached_rows_5525)

@printf("  MAP-OCS:        mean=%.3f s  sd=%.3f s\n", SPRUCE5525_MAP_MEAN,  23.027)
@printf("  CVaR-OCS:       mean=%.3f s  sd=%.3f s\n", SPRUCE5525_CVAR_MEAN, 41.439)
@printf("  CVaR sweep:     %.1f s  (%.1f min) [extrapolated]\n",
        est_sweep_5525, est_sweep_5525/60)
@printf("  Robustness:     %.1f s  (%.1f min) sequential  [extrapolated]\n",
        est_robust_5525, est_robust_5525/60)

for ds in DATASETS
    println("\n" * "-" ^ 70)
    println("DATASET: $(ds.label)")
    println("-" ^ 70)

    # Load or generate GRM
    G = nothing
    if isfile(ds.grm_file)
        println("  Loading GRM: $(ds.grm_file)")
        G, _, n_loaded = load_grm(ds.grm_file)
        if n_loaded != ds.n_expected
            println("  Note: GRM size $n_loaded differs from n_expected $(ds.n_expected) -- using actual size")
        end
        println("  GRM loaded: $n_loaded x $n_loaded  (symmetrised + jitter=1e-6 applied)")
    else
        println("  GRM file not found -- generating synthetic (n=$(ds.n_expected))")
        G = synthetic_grm(ds.n_expected; rng=rng)
    end
    n = size(G, 1)

    # Load or generate g (MAP EBV)
    g = randn(rng, n)
    if !isempty(ds.ebv_file) && isfile(ds.ebv_file)
        ebv_raw = readdlm(ds.ebv_file, ',', header=true)[1]
        g       = Float64.(ebv_raw[:, 2])
        println("  MAP EBV loaded from file")
    else
        println("  Using synthetic MAP EBV")
    end

    # Load or generate U (MCMC scenario matrix, l x n)
    U = nothing
    if !isempty(ds.mcmc_cache) && isfile(ds.mcmc_cache)
        println("  Loading MCMC cache: $(ds.mcmc_cache)")
        @load ds.mcmc_cache U l_scenarios
        println("  MCMC scenarios loaded: $l_scenarios x $n")
    else
        println("  Generating synthetic MCMC scenarios (l=$(L_SCENARIOS))")
        U = synthetic_mcmc_scenarios(g, L_SCENARIOS; rng=rng)
    end

    df_ds = benchmark_dataset(ds.label, G, g, U, ds.theta, ds.n_mcmc_robustness)
    push!(all_results, df_ds)
end

# =============================================================================
# SAVE RESULTS
# =============================================================================

results_df   = vcat(all_results...)
results_file = joinpath(OUT_DIR, "benchmark_results.csv")
CSV.write(results_file, results_df)
println("\nPer-run results saved: $results_file")

# Summary with overhead vs MAP-OCS
summary_df = copy(results_df)
insertcols!(summary_df, :overhead_vs_map => NaN)
for ds_label in unique(summary_df.dataset)
    mask_map = (summary_df.dataset .== ds_label) .& (summary_df.method .== "MAP-OCS")
    t_map    = summary_df[mask_map, :mean_s]
    isempty(t_map) && continue
    mask_ds  = summary_df.dataset .== ds_label
    summary_df[mask_ds, :overhead_vs_map] = summary_df[mask_ds, :mean_s] ./ t_map[1]
end

summary_file = joinpath(OUT_DIR, "benchmark_summary.csv")
CSV.write(summary_file, summary_df)
println("Summary table saved:   $summary_file")

# Console table
println("\n" * "=" ^ 95)
println("BENCHMARK SUMMARY")
println("=" ^ 95)
@printf("  %-20s  %-28s  %6s  %10s  %10s  %8s  %s\n",
        "Dataset", "Method", "n", "Mean (s)", "Mean (min)", "x MAP", "Note")
println("  " * "-" ^ 91)
for row in eachrow(summary_df)
    mean_min     = row.mean_s / 60.0
    overhead_str = isnan(row.overhead_vs_map) ? "  ref" : @sprintf("x%6.1f", row.overhead_vs_map)
    note_str     = ismissing(row.note) ? "" : string(row.note)
    @printf("  %-20s  %-28s  %6d  %10.2f  %10.2f  %s  %s\n",
            row.dataset, row.method, row.n,
            row.mean_s, mean_min, overhead_str, note_str)
end
println("=" ^ 95)
println("""

NOTES FOR MANUSCRIPT:
  MAP-OCS                  timed live; single QP; reference
  CVaR-OCS_single          timed live; single (alpha, mu) solve; l auxiliary vars added
  CVaR-OCS_sweep_estimated extrapolated: 27 x CVaR_single (one-time parameter selection cost)
  RobustnessScore_estimated extrapolated: n_selected x n_mcmc_subset x MAP-OCS
                            (embarrassingly parallel over candidates)
""")
