"""
Re-run Constrained OCS Using Existing Robustness Scores - Norway Spruce
========================================================================
VERSION 2.0: Excludes bottom 25% of MAP-SELECTED individuals

This script:
1. Loads existing robustness scores (already calculated)
2. Runs MAP-OCS on all 1,218 individuals
3. Identifies high-risk individuals (bottom 25% of selected by robustness)
4. Runs Constrained OCS excluding high-risk individuals
5. Calculates contribution distribution metrics for both solutions
6. Saves complete solutions file for visualization

Author: Jon (Skogforsk)
Date: January 2026
"""

using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Printf
using JuMP
using COSMO
using DelimitedFiles: readdlm

println("""
================================================================================
CONSTRAINED OCS WITH EXISTING ROBUSTNESS SCORES
Norway Spruce (Picea abies) - n=1,218 genotyped subset
VERSION 2.0: Excludes bottom 25% of MAP-SELECTED individuals
================================================================================
""")

# ============================================================================
# CONFIGURATION
# ============================================================================

println("\nSTEP 0: Configuration...")

DATA_PATH = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData"
SAVE_PATH = joinpath(DATA_PATH, "Save")

# Files
G_MATRIX_FILE = joinpath(DATA_PATH, "Save", "G_1218_MAF001_mis005_rrBLUP_em_JWAS.txt")
ROBUSTNESS_FILE = joinpath(DATA_PATH, "Save", "robustness_analysis.csv")
BV_FILE = joinpath(DATA_PATH, "Save", "BV_1218.txt")

THETA = 0.02
EXCLUSION_PCT = 0.25  # Exclude bottom 25% of selected individuals by robustness

OUTPUT_FILE = joinpath(SAVE_PATH, "spruce_ocs_solutions_complete.csv")

println("✓ Configuration complete")

# ============================================================================
# STEP 1: Load G Matrix
# ============================================================================

println("\n" * "="^80)
println("STEP 1: Loading G Matrix")
println("="^80)

Amat_t = readdlm(G_MATRIX_FILE, ',', Float64, '\n', header=false)
IDs = Amat_t[:, 1]
G = Amat_t[:, 2:end]

n_individuals = size(G, 1)
println("✓ G matrix loaded")
println("  Individuals: $n_individuals")

# ============================================================================
# STEP 2: Load MAP Breeding Values
# ============================================================================

println("\n" * "="^80)
println("STEP 2: Loading MAP Breeding Values")
println("="^80)

bv_data = readdlm(BV_FILE, ',', Float64, '\n', header=false)
gav = bv_data[:, 2]  # Second column contains breeding values

# Standardize
mean_gav = mean(gav)
var_gav = var(gav)
centered_gav = gav .- mean_gav
n_gav = centered_gav ./ sqrt(var_gav)

println("✓ MAP breeding values loaded and standardized")
println("  Mean: $(round(mean_gav, digits=4))")
println("  Std: $(round(sqrt(var_gav), digits=4))")

# ============================================================================
# STEP 3: Load Robustness Scores
# ============================================================================

println("\n" * "="^80)
println("STEP 3: Loading Robustness Scores")
println("="^80)

robustness_df = CSV.read(ROBUSTNESS_FILE, DataFrame)

println("✓ Robustness scores loaded")
println("  Individuals with scores: $(nrow(robustness_df))")

# Create lookup dictionary: individual_index -> robustness_score
robustness_lookup = Dict(robustness_df.individual .=> robustness_df.robustness_score)

# For individuals without robustness scores, assign NaN
all_robustness = [get(robustness_lookup, i, NaN) for i in 1:n_individuals]

println("  Individuals with scores: $(sum(.!isnan.(all_robustness)))")
println("  Individuals without scores: $(sum(isnan.(all_robustness)))")

# ============================================================================
# STEP 4: Define OCS Function
# ============================================================================

function optimum_contribution(A::Matrix{Float64}, g::Vector{Float64}, 
                              theta::Float64, exclude_indices::Vector{Int}=Int[])
    n_ind = length(g)
    
    model = Model(optimizer_with_attributes(
        COSMO.Optimizer,
        "max_iter" => 10000,
        "eps_abs" => 1e-4,
        "eps_rel" => 1e-4,
        "verbose" => false
    ))
    
    @variable(model, c[1:n_ind] >= 0)
    
    for idx in exclude_indices
        @constraint(model, c[idx] == 0)
    end
    
    @constraint(model, sum(c) == 1)
    @constraint(model, 0.5 * c' * A * c <= theta)
    
    @objective(model, Max, c' * g)
    
    optimize!(model)
    
    return value.(c), termination_status(model)
end

# ============================================================================
# STEP 5: Run MAP-OCS (All Individuals)
# ============================================================================

println("\n" * "="^80)
println("STEP 5: Running MAP-OCS")
println("="^80)

c_map, status_map = optimum_contribution(G, n_gav, THETA)

threshold = 1e-4
selected_indices = findall(c_map .> threshold)
n_selected = length(selected_indices)

gain_map = c_map' * n_gav
coancestry_map = 0.5 * c_map' * G * c_map

println("✓ MAP-OCS complete")
println("  Status: $status_map")
println("  Selected: $n_selected individuals")
println("  Genetic gain: $(round(gain_map, digits=6))")
println("  Coancestry: $(round(coancestry_map, digits=6))")

# ============================================================================
# STEP 6: Identify High-Risk Individuals
# ============================================================================

println("\n" * "="^80)
println("STEP 6: Identifying High-Risk Individuals")
println("="^80)

# Strategy: Exclude bottom 25% of MAP-SELECTED individuals by robustness score
# (Only among those with robustness scores AND selected by MAP-OCS)

# Find selected individuals that have robustness scores
selected_with_scores = intersect(selected_indices, findall(.!isnan.(all_robustness)))
selected_robustness = all_robustness[selected_with_scores]

println("  MAP-OCS selected: $n_selected individuals")
println("  Selected WITH robustness scores: $(length(selected_with_scores))")

if length(selected_with_scores) == 0
    error("No selected individuals have robustness scores! Cannot identify high-risk.")
end

# Calculate threshold: bottom 25% of SELECTED individuals
sorted_selected_scores = sort(selected_robustness)
n_to_exclude = Int(ceil(length(sorted_selected_scores) * EXCLUSION_PCT))
risk_threshold_value = sorted_selected_scores[n_to_exclude]

println("  Exclusion target: Bottom $(EXCLUSION_PCT*100)% = $n_to_exclude individuals")
println("  Risk threshold value: $(round(risk_threshold_value, digits=6))")

# Identify high-risk selected individuals (bottom 25%)
high_risk_selected = selected_with_scores[selected_robustness .<= risk_threshold_value]

println("  High-risk individuals to exclude: $(length(high_risk_selected))")
println("  Percentage of selected: $(round(length(high_risk_selected)/n_selected*100, digits=1))%")

# Show some details
if length(high_risk_selected) > 0
    println("\n  High-risk robustness score range:")
    high_risk_scores = all_robustness[high_risk_selected]
    println("    Min: $(round(minimum(high_risk_scores), digits=6))")
    println("    Max: $(round(maximum(high_risk_scores), digits=6))")
    println("    Mean: $(round(mean(high_risk_scores), digits=6))")
end

# ============================================================================
# STEP 7: Run Constrained OCS (Exclude High-Risk)
# ============================================================================

println("\n" * "="^80)
println("STEP 7: Running Constrained OCS")
println("="^80)

c_constrained, status_const = optimum_contribution(G, n_gav, THETA, high_risk_selected)

constrained_selected_indices = findall(c_constrained .> threshold)
n_constrained = length(constrained_selected_indices)

gain_constrained = c_constrained' * n_gav
coancestry_constrained = 0.5 * c_constrained' * G * c_constrained

gain_loss_pct = (gain_map - gain_constrained) / gain_map * 100

println("✓ Constrained OCS complete")
println("  Status: $status_const")
println("  Selected: $n_constrained individuals")
println("  Genetic gain: $(round(gain_constrained, digits=6))")
println("  Genetic gain loss: $(round(gain_loss_pct, digits=2))%")
println("  Coancestry: $(round(coancestry_constrained, digits=6))")

# ============================================================================
# STEP 8: Calculate Contribution Distribution Metrics
# ============================================================================

println("\n" * "="^80)
println("STEP 8: Calculating Contribution Distribution Metrics")
println("="^80)

# Marginal contributions (c × BV)
map_marginal = c_map .* n_gav
const_marginal = c_constrained .* n_gav

# For selected individuals only
map_selected_contrib = abs.(map_marginal[selected_indices])
const_selected_contrib = abs.(const_marginal[constrained_selected_indices])

# Calculate metrics
function calc_metrics(contributions)
    abs_contrib = abs.(contributions)
    return (
        mean = mean(abs_contrib),
        max = maximum(abs_contrib),
        max_pct = maximum(abs_contrib) / sum(abs_contrib) * 100,
        gini = sum((2 .* (1:length(abs_contrib)) .- length(abs_contrib) .- 1) .* sort(abs_contrib)) / 
               (length(abs_contrib) * sum(abs_contrib))
    )
end

map_metrics = calc_metrics(map_selected_contrib)
const_metrics = calc_metrics(const_selected_contrib)

println("MAP-OCS Contribution Distribution:")
println("  Mean: $(round(map_metrics.mean, digits=6))")
println("  Max: $(round(map_metrics.max, digits=6)) ($(round(map_metrics.max_pct, digits=2))%)")
println("  Gini: $(round(map_metrics.gini, digits=4))")

println("\nConstrained OCS Contribution Distribution:")
println("  Mean: $(round(const_metrics.mean, digits=6))")
println("  Max: $(round(const_metrics.max, digits=6)) ($(round(const_metrics.max_pct, digits=2))%)")
println("  Gini: $(round(const_metrics.gini, digits=4))")

println("\nChanges:")
println("  Mean: $(round((const_metrics.mean - map_metrics.mean)/map_metrics.mean*100, digits=2))%")
println("  Max: $(round((const_metrics.max - map_metrics.max)/map_metrics.max*100, digits=2))%")
println("  Gini: $(round((const_metrics.gini - map_metrics.gini)/map_metrics.gini*100, digits=2))%")

# ============================================================================
# STEP 8b: Calculate Robustness Score Improvement
# ============================================================================

println("\n" * "="^80)
println("STEP 8b: Robustness Score Comparison")
println("="^80)

# Get robustness scores for MAP-OCS selected individuals
map_selected_robustness = Float64[]
for ind in selected_indices
    if !isnan(all_robustness[ind])
        push!(map_selected_robustness, all_robustness[ind])
    end
end

# Get robustness scores for Constrained-OCS selected individuals
const_selected_robustness = Float64[]
for ind in constrained_selected_indices
    if !isnan(all_robustness[ind])
        push!(const_selected_robustness, all_robustness[ind])
    end
end

println("MAP-OCS Selected Individuals:")
println("  Total selected: $n_selected")
println("  With robustness scores: $(length(map_selected_robustness))")
if length(map_selected_robustness) > 0
    println("  Mean robustness: $(round(mean(map_selected_robustness), digits=6))")
    println("  Std dev: $(round(std(map_selected_robustness), digits=6))")
    println("  Range: [$(round(minimum(map_selected_robustness), digits=6)), $(round(maximum(map_selected_robustness), digits=6))]")
end

println("\nConstrained-OCS Selected Individuals:")
println("  Total selected: $n_constrained")
println("  With robustness scores: $(length(const_selected_robustness))")
if length(const_selected_robustness) > 0
    println("  Mean robustness: $(round(mean(const_selected_robustness), digits=6))")
    println("  Std dev: $(round(std(const_selected_robustness), digits=6))")
    println("  Range: [$(round(minimum(const_selected_robustness), digits=6)), $(round(maximum(const_selected_robustness), digits=6))]")
end

# Calculate improvement
if length(map_selected_robustness) > 0 && length(const_selected_robustness) > 0
    robustness_improvement = ((mean(const_selected_robustness) - mean(map_selected_robustness)) / 
                              abs(mean(map_selected_robustness))) * 100
    println("\nRobustness Improvement:")
    println("  Change in mean robustness: $(round(robustness_improvement, digits=2))%")
    
    if robustness_improvement > 0
        println("  ✓ Constrained solution improved mean robustness of selected individuals")
    else
        println("  ✗ Constrained solution did not improve mean robustness")
    end
end

# ============================================================================
# STEP 9: Create Complete Solutions File
# ============================================================================

println("\n" * "="^80)
println("STEP 9: Creating Complete Solutions File")
println("="^80)

# Create DataFrame with ALL 1,218 individuals
solutions_df = DataFrame(
    individual = 1:n_individuals,
    individual_id = Int.(IDs),
    map_contribution = c_map,
    map_marginal_contrib = map_marginal,
    constrained_contribution = c_constrained,
    constrained_marginal_contrib = const_marginal,
    breeding_value = gav,
    standardized_bv = n_gav,
    robustness_score = all_robustness,
    was_excluded = [i in high_risk_selected for i in 1:n_individuals],
    selected_in_map = [i in selected_indices for i in 1:n_individuals],
    selected_in_constrained = [i in constrained_selected_indices for i in 1:n_individuals]
)

CSV.write(OUTPUT_FILE, solutions_df)

println("✓ Complete solutions file saved")
println("  File: $OUTPUT_FILE")
println("  Rows: $(nrow(solutions_df))")
println("  All 1,218 individuals included")

# ============================================================================
# STEP 10: Summary
# ============================================================================

println("\n" * "="^80)
println("SUMMARY - NORWAY SPRUCE DUAL METRICS")
println("="^80)

println("\nGENETIC GAIN & SELECTION:")
println("  MAP-OCS gain: $(round(gain_map, digits=6))")
println("  Constrained gain: $(round(gain_constrained, digits=6))")
println("  Loss: $(round(gain_loss_pct, digits=2))%")
println("  MAP selected: $n_selected")
println("  Constrained selected: $n_constrained")

println("\nHIGH-RISK INDIVIDUALS:")
println("  Identified: $(length(high_risk_selected))")
println("  Excluded in Constrained: $(length(high_risk_selected))")

println("\nCONTRIBUTION DISTRIBUTION:")
println("  Mean contribution change: $(round((const_metrics.mean - map_metrics.mean)/map_metrics.mean*100, digits=2))%")
println("  Max contribution change: $(round((const_metrics.max - map_metrics.max)/map_metrics.max*100, digits=2))%")

println("\n" * "="^80)
println("COMPLETE! Now ready for visualization")
println("="^80)
println("\nNext step:")
println("  Use visualize_dual_metrics_spruce_v2.jl with:")
println("  ROBUSTNESS_FILE = robustness_analysis.csv")
println("  SOLUTIONS_FILE = spruce_ocs_solutions_complete.csv")
println("="^80)
