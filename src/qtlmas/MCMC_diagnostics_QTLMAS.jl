# ============================================================================
# MCMC-OCS DIAGNOSTIC — QTL-MAS 2010 (run after cvar_ocs_qtlmas2010_5.jl)
# Overlap metrics + selection-frequency analysis vs MAP-OCS, Θ = 0.03
# Reuses: G, U, g_map, c_map, sire_idx, dam_idx, run_map_ocs, THETA (=0.03)
# ============================================================================

using Statistics, StatsBase

n_cand = length(g_map)
map_selected = Set(findall(c_map .> 1e-4))
n_map = length(map_selected)

overlaps          = Vector{Int}(undef, l_scenarios)
jaccards          = Vector{Float64}(undef, l_scenarios)
rank_corrs        = Vector{Float64}(undef, l_scenarios)
selection_counts  = zeros(Int, n_cand)   # for selection-frequency analysis
n_selected_mcmc   = Vector{Int}(undef, l_scenarios)

println("Running per-scenario MCMC-OCS ($l_scenarios iterations, sex-constrained, Θ=$THETA)...")

for j in 1:l_scenarios
    g_j = U[:, j]  # per-draw standardised index, exactly as used for CVaR-OCS scenarios

    c_j, gain_j, coanc_j, status_j = run_map_ocs(G, g_j, THETA; sire_idx=sire_idx, dam_idx=dam_idx)

    mcmc_selected = Set(findall(c_j .> 1e-4))
    n_selected_mcmc[j] = length(mcmc_selected)

    overlap_set = intersect(map_selected, mcmc_selected)
    union_set   = union(map_selected, mcmc_selected)
    overlaps[j] = length(overlap_set)
    jaccards[j] = length(overlap_set) / length(union_set)
    rank_corrs[j] = corspearman(c_map, c_j)

    for idx in mcmc_selected
        selection_counts[idx] += 1
    end

    if j % 100 == 0
        println("  Processed $j / $l_scenarios iterations")
    end
end

selection_freq = selection_counts ./ l_scenarios

# ── Summary statistics matching the spruce presentation ─────────────────────
println("\n" * "="^70)
println("MCMC-OCS DIAGNOSTIC SUMMARY — QTL-MAS 2010 (Θ = $THETA)")
println("="^70)
@printf("  Mean overlap with MAP-OCS   : %.1f  (range: %d--%d)\n",
        mean(overlaps), minimum(overlaps), maximum(overlaps))
@printf("  Mean Jaccard similarity     : %.1f%%\n", mean(jaccards)*100)
@printf("  Max selection frequency     : %.1f%%\n", maximum(selection_freq)*100)

# Rank-order relationship between EBV and selection frequency
spearman_rho = corspearman(g_map, selection_freq)
@printf("  Spearman rho (EBV vs freq)  : %.3f\n", spearman_rho)

# Quartile ratio: top-25% EBV vs bottom-25% EBV mean selection frequency
q75 = quantile(g_map, 0.75)
q25 = quantile(g_map, 0.25)
top_quartile_freq    = mean(selection_freq[g_map .>= q75])
bottom_quartile_freq = mean(selection_freq[g_map .<= q25])
quartile_ratio = top_quartile_freq / bottom_quartile_freq
@printf("  Top-quartile / bottom-quartile selection frequency ratio: %.1fx\n", quartile_ratio)

println("="^70)

# Save for verification / figure use
using CSV, DataFrames
mcmc_ocs_df = DataFrame(
    individual = 1:n_cand,
    ebv_index = g_map,
    selection_frequency = selection_freq
)
theta_str = "Theta$(Int(round(THETA*100)))"  # e.g. Theta3 for 0.03
mkpath(joinpath(SAVE_DIR, theta_str))
CSV.write(joinpath(SAVE_DIR, theta_str, "mcmc_ocs_diagnostic_selection_freq.csv"), mcmc_ocs_df)


overlap_df = DataFrame(
    iteration = 1:l_scenarios,
    overlap = overlaps,
    jaccard = jaccards,
    rank_corr = rank_corrs,
    n_selected = n_selected_mcmc
)
CSV.write(joinpath(SAVE_DIR, theta_str, "mcmc_ocs_diagnostic_overlap.csv"), overlap_df)