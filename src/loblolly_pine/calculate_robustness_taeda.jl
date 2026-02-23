"""
MCMC-Based Individual Robustness Score Calculation - Loblolly Pine
==================================================================

Calculates robustness scores for top 200 individuals in Loblolly pine
by re-optimizing OCS with each individual excluded.

This generates the data needed for METRIC 1 (Individual Importance)

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
using Random

println("""
================================================================================
MCMC-BASED ROBUSTNESS SCORE CALCULATION
Loblolly Pine (Pinus taeda) - Top 200 Individuals
================================================================================
""")

# ============================================================================
# CONFIGURATION
# ============================================================================

println("\nSTEP 0: Configuration...")

DATA_PATH = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData"
RESULTS_PATH = joinpath(DATA_PATH, "results_G_926")

# MCMC files
MCMC_FILES = Dict(
    "HT6" => joinpath(RESULTS_PATH, "MCMC_samples_EBV_HT6.txt"),
    "DBH6" => joinpath(RESULTS_PATH, "MCMC_samples_EBV_DBH6.txt"),
    "GV6" => joinpath(RESULTS_PATH, "MCMC_samples_EBV_GV6.txt"),
    "WDN4" => joinpath(RESULTS_PATH, "MCMC_samples_EBV_WDN4.txt")
)

# EBV files
EBV_FILES = Dict(
    "HT6" => joinpath(RESULTS_PATH, "EBV_HT6.txt"),
    "DBH6" => joinpath(RESULTS_PATH, "EBV_DBH6.txt"),
    "GV6" => joinpath(RESULTS_PATH, "EBV_GV6.txt"),
    "WDN4" => joinpath(RESULTS_PATH, "EBV_WDN4.txt")
)

G_MATRIX_FILE = joinpath(DATA_PATH, "G_926_MAF001_mis005_rrBLUP_em_JWAS.txt")

THETA = 0.05
N_TOP_ANALYZE = 4  # Top individuals to analyze
N_MCMC_SAMPLES = 5  # Number of MCMC samples to use (set low for debugging)

OUTPUT_FILE = joinpath(DATA_PATH, "stable_robustness_results_taeda.csv")

println("✓ Configuration complete")
println("  Analyzing top $N_TOP_ANALYZE individuals")
println("  Using $N_MCMC_SAMPLES MCMC samples")

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
# STEP 2: Load MCMC and Calculate Index
# ============================================================================

println("\n" * "="^80)
println("STEP 2: Loading MCMC Breeding Values")
println("="^80)

mcmc_HT6 = CSV.read(MCMC_FILES["HT6"], DataFrame, delim=',', header=true, missingstring="NA")
mcmc_DBH6 = CSV.read(MCMC_FILES["DBH6"], DataFrame, delim=',', header=true, missingstring="NA")
mcmc_GV6 = CSV.read(MCMC_FILES["GV6"], DataFrame, delim=',', header=true, missingstring="NA")
mcmc_WDN4 = CSV.read(MCMC_FILES["WDN4"], DataFrame, delim=',', header=true, missingstring="NA")

println("✓ MCMC samples loaded")

# Convert to matrices
mcmc_ht6_mat = Matrix(mcmc_HT6)
mcmc_dbh6_mat = Matrix(mcmc_DBH6)
mcmc_gv6_mat = Matrix(mcmc_GV6)
mcmc_wdn4_mat = Matrix(mcmc_WDN4)

# Calculate index: (HT6 + DBH6 + WDN4 - GV6) / 4
println("\nCalculating index: (HT6 + DBH6 + WDN4 - GV6) / 4")
mcmc_matrix = (mcmc_ht6_mat .+ mcmc_dbh6_mat .+ mcmc_wdn4_mat .- mcmc_gv6_mat) ./ 4

n_iterations = size(mcmc_matrix, 1)
println("✓ Index calculated")
println("  MCMC iterations: $n_iterations")

# ============================================================================
# STEP 3: Calculate MAP EBVs
# ============================================================================

println("\n" * "="^80)
println("STEP 3: Calculating MAP Breeding Values")
println("="^80)

H6 = CSV.read(EBV_FILES["HT6"], DataFrame)
DBH6 = CSV.read(EBV_FILES["DBH6"], DataFrame)
GV6 = CSV.read(EBV_FILES["GV6"], DataFrame)
WDN4 = CSV.read(EBV_FILES["WDN4"], DataFrame)

gav = zeros(Float64, n_individuals)
for i in 1:n_individuals
    gav[i] = 0.25 * (DBH6[i, 2] + H6[i, 2] + WDN4[i, 2] - GV6[i, 2])
end

# Standardize
mean_gav = mean(gav)
var_gav = var(gav)
centered_gav = gav .- mean_gav
n_gav = centered_gav ./ sqrt(var_gav)

println("✓ MAP breeding values calculated and standardized")

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
# STEP 5: Run MAP-OCS to Identify Top Individuals
# ============================================================================

println("\n" * "="^80)
println("STEP 5: Running MAP-OCS")
println("="^80)

c_map, status = optimum_contribution(G, n_gav, THETA)

threshold = 1e-4
selected_indices = findall(c_map .> threshold)
n_selected = length(selected_indices)

gain_map = c_map' * n_gav

println("✓ MAP-OCS complete")
println("  Status: $status")
println("  Selected: $n_selected individuals")
println("  Genetic gain: $(round(gain_map, digits=6))")

# Identify individuals to analyze: ALL selected + top unselected
println("\n  Identifying analysis set...")

# All selected individuals
selected_set = selected_indices

# TEMPORARILY: Only analyze selected individuals for debugging
top_indices = selected_set
n_analyze = length(top_indices)

println("  Analysis set (DEBUG MODE - selected only):")
println("    - Selected individuals: $n_selected")
println("    - Total to analyze: $n_analyze")

# ============================================================================
# STEP 6: Calculate Robustness Scores
# ============================================================================

println("\n" * "="^80)
println("STEP 6: Calculating MCMC-Based Robustness Scores")
println("="^80)

println("\nThis will take some time...")
println("  Individuals to analyze: $n_analyze")
println("  MCMC samples: $N_MCMC_SAMPLES")
println("  Total optimizations: $(n_analyze * N_MCMC_SAMPLES)")

# Sample MCMC iterations
sample_indices = rand(1:n_iterations, N_MCMC_SAMPLES)

# Storage
robustness_scores = zeros(Float64, n_analyze)
robustness_matrix = zeros(Float64, N_MCMC_SAMPLES, n_analyze)

println("\nProcessing individuals...")

for (idx, individual) in enumerate(top_indices)
    if idx % 10 == 0 || idx == 1
        println("  Individual $idx/$n_analyze (ID: $(Int(IDs[individual])))")
    end
    
    gains_without = Float64[]
    gains_with = Float64[]  # Track baseline WITH individual
    
    for (iter_idx, mcmc_iter) in enumerate(sample_indices)
        # Get breeding values for this iteration
        iteration_bvs = mcmc_matrix[mcmc_iter, :]
        mean_iter = mean(iteration_bvs)
        var_iter = var(iteration_bvs)
        centered_iter = iteration_bvs .- mean_iter
        n_iter = centered_iter ./ sqrt(var_iter)
        
        # Baseline WITH individual (no exclusions)
        c_with, status_with = optimum_contribution(G, n_iter, THETA)
        
        gain_with = 0.0  # Initialize outside if block
        if status_with == :OPTIMAL
            gain_with = c_with' * n_iter
            push!(gains_with, gain_with)
        else
            gain_with = gain_map
            push!(gains_with, gain_map)
        end
        
        # Re-optimize WITHOUT this individual
        c_without, status_without = optimum_contribution(G, n_iter, THETA, [individual])
        
        gain_without = 0.0  # Initialize outside if block
        if status_without == :OPTIMAL
            gain_without = c_without' * n_iter
            push!(gains_without, gain_without)
        else
            # If optimization fails, use a penalty
            gain_without = gain_map * 0.9
            push!(gains_without, gain_map * 0.9)
        end
        
        # Debug first individual's first few iterations
        if idx == 1 && iter_idx <= 3
            println("    Iteration $iter_idx:")
            println("      gain_with: $gain_with")
            println("      gain_without: $gain_without")
            println("      difference: $(gain_with - gain_without)")
            println("      c_with[individual=$individual]: $(c_with[individual])")
            println("      c_without[individual=$individual]: $(c_without[individual])")
        end
    end
    
    # Calculate robustness score = mean genetic gain loss when individual excluded
    mean_gain_with = mean(gains_with)
    mean_gain_without = mean(gains_without)
    robustness_scores[idx] = mean_gain_with - mean_gain_without
    
    # Debug first few individuals
    if idx <= 3
        println("  Individual $idx summary:")
        println("    mean_gain_with: $mean_gain_with")
        println("    mean_gain_without: $mean_gain_without")
        println("    robustness: $(robustness_scores[idx])")
        println("    n_samples: $(length(gains_with))")
    end
    
    # Store all gains for variability analysis
    robustness_matrix[:, idx] = gains_without
end

println("\n✓ Robustness scores calculated!")

# ============================================================================
# STEP 7: Create Results DataFrame
# ============================================================================

println("\n" * "="^80)
println("STEP 7: Creating Results DataFrame")
println("="^80)

# Track which individuals were originally selected
selected_status = [i in selected_indices for i in top_indices]

results_df = DataFrame(
    individual = top_indices,
    individual_id = Int.(IDs[top_indices]),
    selected_in_map = selected_status,
    map_contribution = c_map[top_indices],
    breeding_value = gav[top_indices],
    robustness_score = robustness_scores,
    percentage_impact = (robustness_scores ./ gain_map) .* 100,
    standardized_robustness = (robustness_scores .- mean(robustness_scores)) ./ std(robustness_scores),
    max_scaled_robustness = robustness_scores ./ maximum(robustness_scores)
)

# Add quartile classification
q25 = quantile(robustness_scores, 0.25)
q50 = quantile(robustness_scores, 0.50)
q75 = quantile(robustness_scores, 0.75)

results_df.importance_quartile = map(results_df.robustness_score) do score
    if score < q25
        "Q1 (Lowest Importance)"
    elseif score < q50
        "Q2 (Low-Moderate)"
    elseif score < q75
        "Q3 (Moderate-High)"
    else
        "Q4 (Highest Importance)"
    end
end

println("✓ Results DataFrame created")
println("\nRobustness Score Statistics:")
println("  Mean: $(round(mean(robustness_scores), digits=6))")
println("  Median: $(round(median(robustness_scores), digits=6))")
println("  Range: [$(round(minimum(robustness_scores), digits=6)), $(round(maximum(robustness_scores), digits=6))]")
println("  Q1: $(round(q25, digits=6))")
println("  Q3: $(round(q75, digits=6))")

# Summary by selection status
selected_rob = robustness_scores[selected_status]
unselected_rob = robustness_scores[.!selected_status]

println("\nBy Selection Status:")
println("  Selected (n=$(sum(selected_status))):")
println("    Mean robustness: $(round(mean(selected_rob), digits=6))")
println("    Range: [$(round(minimum(selected_rob), digits=6)), $(round(maximum(selected_rob), digits=6))]")

if length(unselected_rob) > 0
    println("  Unselected (n=$(sum(.!selected_status))):")
    println("    Mean robustness: $(round(mean(unselected_rob), digits=6))")
    println("    Range: [$(round(minimum(unselected_rob), digits=6)), $(round(maximum(unselected_rob), digits=6))]")
end

# ============================================================================
# STEP 8: Save Results
# ============================================================================

println("\n" * "="^80)
println("STEP 8: Saving Results")
println("="^80)

CSV.write(OUTPUT_FILE, results_df)
println("✓ Saved: $OUTPUT_FILE")

println("\n" * "="^80)
println("ANALYSIS COMPLETE!")
println("="^80)
println("\nRobustness scores calculated for top $n_analyze Loblolly pine individuals")
println("Output file: $OUTPUT_FILE")
println("\nNext step:")
println("  Run dual_metrics_taeda_pine.jl again to use these scores!")
println("="^80)
