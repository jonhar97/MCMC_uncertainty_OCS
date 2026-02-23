"""
Working Example: MCMC-OCS Uncertainty Analysis
===============================================

Demonstrates the full uncertainty-aware OCS pipeline on simulated data.
Runtime: ~3-5 minutes on a standard laptop.

Steps:
  1. Load simulated data (or generate it first)
  2. Run MAP-OCS (classical point-estimate approach)
  3. Run MCMC-OCS (one optimisation per posterior draw)
  4. Calculate individual robustness scores
  5. Run Constrained OCS (excluding high-risk individuals)
  6. Compare and visualise results

Author: Jon Ahlinder (Skogforsk) / Patrik Waldmann

USAGE:
  # Generate data first (only needed once):
  include("simulate_example_data.jl")

  # Then run this example:
  include("run_example.jl")
"""

using CSV, DataFrames, DelimitedFiles
using Statistics, LinearAlgebra, Printf
using JuMP, COSMO
using Plots, StatsPlots

EXAMPLE_PATH = @__DIR__

println("="^70)
println("MCMC-OCS UNCERTAINTY ANALYSIS — WORKING EXAMPLE")
println("="^70)

# ============================================================================
# CONFIGURATION
# ============================================================================
THETA          = 0.02     # Coancestry constraint
CONTRIB_THRESH = 1e-4     # Minimum contribution to be counted as "selected"
EXCLUSION_PCT  = 0.25     # Bottom 25% of selected by robustness → excluded
N_ROBUSTNESS   = 50       # Number of top individuals to calculate robustness for

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
println("\nSTEP 1: Loading data...")

# Check files exist
required = ["example_G_matrix.txt", "example_MCMC_EBV_samples.txt", "example_MAP_EBV.csv"]
for f in required
    fpath = joinpath(EXAMPLE_PATH, f)
    if !isfile(fpath)
        error("Missing: $fpath\nPlease run simulate_example_data.jl first.")
    end
end

# G matrix
println("  Loading G matrix...")
G_raw  = readdlm(joinpath(EXAMPLE_PATH, "example_G_matrix.txt"), ',', Float64)
IDs    = Int.(G_raw[:, 1])
G      = G_raw[:, 2:end]
n_ind  = size(G, 1)
println("  ✔ G matrix: $n_ind × $n_ind individuals")

# MCMC EBV samples (individuals × iterations)
println("  Loading MCMC EBV samples...")
mcmc_samples = readdlm(joinpath(EXAMPLE_PATH, "example_MCMC_EBV_samples.txt"), ',', Float64)
n_mcmc       = size(mcmc_samples, 2)
println("  ✔ MCMC samples: $n_ind individuals × $n_mcmc iterations")

# MAP EBVs and metadata
println("  Loading MAP EBVs...")
map_df  = CSV.read(joinpath(EXAMPLE_PATH, "example_MAP_EBV.csv"), DataFrame)
map_ebv = map_df.MAP_EBV
family  = map_df.Family
println("  ✔ MAP EBVs: range [$(round(minimum(map_ebv),digits=2)), $(round(maximum(map_ebv),digits=2))]")

# Standardise EBVs (match analysis scripts)
map_ebv_std = (map_ebv .- mean(map_ebv)) ./ std(map_ebv)
mcmc_std    = (mcmc_samples .- mean(map_ebv)) ./ std(map_ebv)

# ============================================================================
# STEP 2: OCS SOLVER
# ============================================================================

"""
Solve the OCS quadratic programme.

Maximises  c'g  subject to:
  0.5 * c'Gc ≤ θ   (coancestry constraint)
  sum(c) = 1        (contributions sum to 1)
  c ≥ 0             (non-negative)
  c[exclude] = 0    (excluded individuals)
"""
function solve_ocs(G::Matrix{Float64}, g::Vector{Float64},
                   theta::Float64; exclude::Vector{Int}=Int[])
    n = length(g)
    model = Model(optimizer_with_attributes(
        COSMO.Optimizer,
        "max_iter" => 8000,
        "eps_abs"  => 1e-5,
        "eps_rel"  => 1e-5,
        "verbose"  => false
    ))
    @variable(model, c[1:n] >= 0)
    for i in exclude
        @constraint(model, c[i] == 0)
    end
    @constraint(model, sum(c) == 1)
    @constraint(model, 0.5 * c' * G * c <= theta)
    @objective(model, Max, c' * g)
    optimize!(model)
    status = termination_status(model)
    if status == MOI.OPTIMAL || status == MOI.ALMOST_OPTIMAL
        return value.(c), termination_status(model)
    else
        return zeros(n), status
    end
end

# ============================================================================
# STEP 3: MAP-OCS (classical)
# ============================================================================
println("\nSTEP 2: Running MAP-OCS (classical point estimate)...")

c_map, status_map = solve_ocs(G, map_ebv_std, THETA)
selected_map      = findall(c_map .> CONTRIB_THRESH)
gain_map          = c_map' * map_ebv_std
coancestry_map    = 0.5 * c_map' * G * c_map

println("  Status    : $status_map")
println("  Selected  : $(length(selected_map)) individuals")
@printf("  Gain      : %.4f\n", gain_map)
@printf("  Coancestry: %.4f (limit = %.4f)\n", coancestry_map, THETA)

# ============================================================================
# STEP 4: MCMC-OCS
# ============================================================================
println("\nSTEP 3: Running MCMC-OCS ($n_mcmc iterations)...")
println("  (This is the computationally intensive step)")

contributions_mcmc = zeros(n_ind, n_mcmc)   # individual × iteration
gains_mcmc         = zeros(n_mcmc)
n_failed           = 0

for iter in 1:n_mcmc
    if iter % 100 == 0
        println("    Iteration $iter / $n_mcmc  ($(n_failed) failed so far)")
    end
    g_iter = mcmc_std[:, iter]
    c_iter, status = solve_ocs(G, g_iter, THETA)
    if status == MOI.OPTIMAL || status == MOI.ALMOST_OPTIMAL
        contributions_mcmc[:, iter] = c_iter
        gains_mcmc[iter]            = c_iter' * map_ebv_std  # evaluate on MAP EBVs
    else
        n_failed += 1
    end
end

n_valid = n_mcmc - n_failed
println("  ✔ Completed: $n_valid / $n_mcmc iterations successful")

# Selection frequencies (proportion of successful iterations each individual was selected)
valid_iters     = findall(vec(sum(contributions_mcmc, dims=1)) .> 0)
sel_freq        = vec(sum(contributions_mcmc[:, valid_iters] .> CONTRIB_THRESH, dims=2)) ./ length(valid_iters)

println("  Individuals selected in ≥50% of iterations: $(sum(sel_freq .>= 0.5))")
println("  Individuals never selected               : $(sum(sel_freq .== 0))")

# ============================================================================
# STEP 5: ROBUSTNESS SCORES
# ============================================================================
println("\nSTEP 4: Calculating individual robustness scores...")
println("  (Top $N_ROBUSTNESS candidates by MAP contribution)")

# Candidates: top N individuals by MAP contribution
candidate_idx = sortperm(c_map, rev=true)[1:N_ROBUSTNESS]

# Gain with full candidate pool (MAP EBVs for comparability)
c_full, _   = solve_ocs(G, map_ebv_std, THETA)
gain_full   = c_full' * map_ebv_std

robustness_scores = fill(NaN, n_ind)

for (k, ind) in enumerate(candidate_idx)
    k % 10 == 0 && println("    $k / $N_ROBUSTNESS")
    c_excl, status = solve_ocs(G, map_ebv_std, THETA, exclude=[ind])
    if status == MOI.OPTIMAL || status == MOI.ALMOST_OPTIMAL
        gain_excl              = c_excl' * map_ebv_std
        robustness_scores[ind] = gain_excl / gain_full
    end
end

scored     = findall(.!isnan.(robustness_scores))
n_scored   = length(scored)
println("  ✔ Scored $n_scored individuals")
println("  Robustness range: [$(round(minimum(robustness_scores[scored]),digits=4)), $(round(maximum(robustness_scores[scored]),digits=4))]")
println("  (1.0 = fully replaceable; <1 = critical dependency)")

# ============================================================================
# STEP 6: CONSTRAINED OCS (exclude high-risk individuals)
# ============================================================================
println("\nSTEP 5: Running Constrained OCS...")

# High-risk = bottom 25% of MAP-SELECTED individuals by robustness
selected_scored = intersect(selected_map, scored)
sel_rob         = robustness_scores[selected_scored]
n_exclude       = max(1, Int(ceil(length(sel_rob) * EXCLUSION_PCT)))
rob_threshold   = sort(sel_rob)[n_exclude]
high_risk       = selected_scored[sel_rob .<= rob_threshold]

println("  MAP-selected with scores: $(length(selected_scored))")
println("  High-risk (bottom 25%)  : $(length(high_risk))")

c_const, status_const = solve_ocs(G, map_ebv_std, THETA, exclude=high_risk)
selected_const         = findall(c_const .> CONTRIB_THRESH)
gain_const             = c_const' * map_ebv_std
coancestry_const       = 0.5 * c_const' * G * c_const
gain_loss_pct          = (gain_map - gain_const) / abs(gain_map) * 100

println("  Status    : $status_const")
println("  Selected  : $(length(selected_const)) individuals")
@printf("  Gain      : %.4f  (%.2f%% loss vs MAP-OCS)\n", gain_const, gain_loss_pct)
@printf("  Coancestry: %.4f\n", coancestry_const)

# Robustness improvement
map_rob_mean   = mean(robustness_scores[intersect(selected_map,  scored)])
const_rob_mean = mean(robustness_scores[intersect(selected_const, scored)])
rob_improvement = (const_rob_mean - map_rob_mean) / abs(map_rob_mean) * 100

@printf("  Mean robustness MAP-OCS       : %.4f\n", map_rob_mean)
@printf("  Mean robustness Constrained   : %.4f\n", const_rob_mean)
@printf("  Robustness improvement        : %.1f%%\n", rob_improvement)

# ============================================================================
# STEP 7: FIGURES
# ============================================================================
println("\nSTEP 6: Generating figures...")

# --- Figure 1: EBV vs Selection Frequency with quadratic fit ---
X_quad  = hcat(ones(n_ind), map_ebv_std, map_ebv_std.^2)
β_quad  = X_quad \ sel_freq
y_fit   = X_quad * β_quad
r2_quad = 1 - sum((sel_freq .- y_fit).^2) / sum((sel_freq .- mean(sel_freq)).^2)

x_grid  = range(minimum(map_ebv_std)-0.2, maximum(map_ebv_std)+0.2, length=200)
X_grid  = hcat(ones(200), collect(x_grid), collect(x_grid).^2)
y_grid  = X_grid * β_quad

p1 = scatter(map_ebv_std, sel_freq,
    alpha=0.5, markersize=4, color=:steelblue, markerstrokewidth=0,
    xlabel="Standardised MAP EBV",
    ylabel="Selection frequency (MCMC-OCS)",
    title="EBV vs Selection Frequency\n(R² = $(round(r2_quad, digits=3)))",
    label="Individuals", legend=:topleft, dpi=150)
plot!(p1, x_grid, y_grid, color=:firebrick, linewidth=2.5,
    label=@sprintf("Quadratic fit (R²=%.3f)", r2_quad))

# --- Figure 2: Robustness scores for MAP-selected individuals ---
sel_rob_vals  = robustness_scores[selected_scored]
sort_order    = sortperm(sel_rob_vals)
rob_colors    = [v <= rob_threshold ? :firebrick : :steelblue for v in sel_rob_vals[sort_order]]

p2 = bar(1:length(sort_order), sel_rob_vals[sort_order],
    color=rob_colors, linecolor=:transparent,
    xlabel="Rank (by robustness score)",
    ylabel="Robustness score",
    title="Individual Robustness Scores\n(red = high-risk, excluded in Constrained OCS)",
    legend=false, dpi=150)
hline!(p2, [rob_threshold], color=:black, linestyle=:dash, linewidth=1.5)

# --- Figure 3: Selection frequency distributions ---
mcmc_sel_freq = sel_freq[selected_map]
map_selected_fam = [family[i] for i in selected_map]

p3 = histogram(mcmc_sel_freq, bins=20, color=:steelblue, alpha=0.7,
    xlabel="MCMC Selection Frequency",
    ylabel="Count",
    title="Selection Frequency Distribution\n(MAP-OCS selected individuals)",
    legend=false, dpi=150)

# --- Figure 4: Summary comparison bar chart ---
labels      = ["MAP-OCS", "Constrained OCS"]
gains       = [gain_map, gain_const]
n_selected  = [length(selected_map), length(selected_const)]
rob_means   = [map_rob_mean, const_rob_mean]

p4a = bar(labels, gains, color=[:steelblue, :seagreen], alpha=0.85,
    ylabel="Genetic gain (standardised)",
    title="Genetic Gain", legend=false, dpi=150)

p4b = bar(labels, rob_means, color=[:steelblue, :seagreen], alpha=0.85,
    ylabel="Mean robustness score",
    title="Mean Robustness\n(selected individuals)", legend=false, dpi=150)

# Combine
fig_combined = plot(p1, p2, p3, plot(p4a, p4b, layout=(1,2)),
    layout=(2,2), size=(1200, 900),
    plot_title="MCMC-OCS Uncertainty Analysis — Working Example")

savefig(fig_combined, joinpath(EXAMPLE_PATH, "example_results.pdf"))
println("  ✔ Saved example_results.pdf")

# ============================================================================
# SUMMARY
# ============================================================================
println()
println("="^70)
println("RESULTS SUMMARY")
println("="^70)
@printf("\n%-35s %10s %10s\n", "Metric", "MAP-OCS", "Constrained")
println("-"^57)
@printf("%-35s %10.4f %10.4f\n", "Genetic gain (standardised)", gain_map, gain_const)
@printf("%-35s %10.4f %10.4f\n", "Coancestry", coancestry_map, coancestry_const)
@printf("%-35s %10d %10d\n",     "Individuals selected", length(selected_map), length(selected_const))
@printf("%-35s %10s %10.1f%%\n", "Gain loss",  "—", gain_loss_pct)
@printf("%-35s %10.4f %10.4f\n", "Mean robustness (selected)", map_rob_mean, const_rob_mean)
@printf("%-35s %10s %10.1f%%\n", "Robustness improvement", "—", rob_improvement)
println("-"^57)
@printf("\nMCMC-OCS EBV-frequency relationship: R² = %.3f (quadratic fit)\n", r2_quad)
println()
println("Figure saved: example_results.pdf")
println()
println("="^70)
println("Done! See the manuscript for interpretation of these metrics.")
println("="^70)
