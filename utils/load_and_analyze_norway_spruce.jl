"""
Load Norway Spruce MCMC Contributions Matrix and Analyze
=========================================================

This script:
1. Loads your mcmc_contributions_matrix.csv
2. Calculates selection frequencies (top 100 per iteration)
3. Runs polynomial and GP regression analysis
4. Generates publication-ready results

USAGE:
    include("load_and_analyze_norway_spruce.jl")
    results = analyze_norway_spruce_mcmc_data(n_gav, mcmc_file_path)
"""

using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Plots
using Printf

# ============================================================================
# LOAD AND PROCESS MCMC CONTRIBUTIONS
# ============================================================================

"""
Load MCMC contributions matrix and calculate selection frequencies

Parameters:
- filepath: Path to mcmc_contributions_matrix.csv
- top_n: Number of top individuals to select per iteration (default: 100)

Returns:
- selection_freq: Vector of selection frequencies (proportion of iterations selected)
- mcmc_matrix: Full contributions matrix for diagnostics
"""
function load_mcmc_contributions(filepath::String; top_n::Int=100)
    println("="^80)
    println("LOADING MCMC CONTRIBUTIONS MATRIX")
    println("="^80)
    
    # Load CSV
    println("\nLoading: $filepath")
    df = CSV.read(filepath, DataFrame)
    
    println("âœ“ Loaded successfully")
    println("  Dimensions: $(size(df))")
    println("  Individuals (rows): $(nrow(df))")
    println("  Iterations (columns): $(ncol(df))")
    
    # Convert to matrix (assuming first column might be individual IDs)
    # Check if first column is IDs or data
    first_col_name = names(df)[1]
    
    if first_col_name in ["Individual", "ID", "IndividualID", "ind", "id"]
        println("  First column detected as IDs: $first_col_name")
        individual_ids = df[!, 1]
        mcmc_matrix = Matrix(df[!, 2:end])
    else
        println("  All columns are MCMC iterations")
        individual_ids = 1:nrow(df)
        mcmc_matrix = Matrix(df)
    end
    
    n_individuals = size(mcmc_matrix, 1)
    n_iterations = size(mcmc_matrix, 2)
    
    println("\nâœ“ Matrix processed:")
    println("  Individuals: $n_individuals")
    println("  MCMC iterations: $n_iterations")
    println("  Top-n per iteration: $top_n")
    
    # Calculate selection frequencies
    println("\nCalculating selection frequencies...")
    selection_freq = zeros(Float64, n_individuals)
    
    for iter in 1:n_iterations
        # Get contributions for this iteration
        contributions = mcmc_matrix[:, iter]
        
        # Find top_n individuals by contribution
        # Get indices sorted by contribution (descending)
        sorted_indices = sortperm(contributions, rev=true)
        top_indices = sorted_indices[1:min(top_n, length(sorted_indices))]
        
        # Increment selection count for these individuals
        selection_freq[top_indices] .+= 1.0
    end
    
    # Convert counts to frequencies (proportion of iterations)
    selection_freq ./= n_iterations
    
    println("âœ“ Selection frequencies calculated")
    println("  Range: [$(round(minimum(selection_freq), digits=3)), $(round(maximum(selection_freq), digits=3))]")
    println("  Mean: $(round(mean(selection_freq), digits=3))")
    println("  Std: $(round(std(selection_freq), digits=3))")
    println("  Individuals selected at least once: $(sum(selection_freq .> 0))")
    println("  Individuals never selected: $(sum(selection_freq .== 0))")
    
    return selection_freq, mcmc_matrix, individual_ids
end

# ============================================================================
# COMPLETE ANALYSIS PIPELINE
# ============================================================================

"""
Complete Norway Spruce analysis pipeline

Parameters:
- n_gav: MAP EBV index values (you already have this loaded)
- mcmc_file_path: Path to mcmc_contributions_matrix.csv
- top_n: Number of top individuals per iteration (default: 100)
- output_file: Where to save results plot

Returns:
- Dictionary with all analysis results
"""
function analyze_norway_spruce_mcmc_data(
    n_gav::Vector{Float64},
    mcmc_file_path::String;
    top_n::Int=100,
    output_file::String="norway_spruce_complete_analysis.png"
)
    # Load and process MCMC data
    selection_freq, mcmc_matrix, individual_ids = load_mcmc_contributions(
        mcmc_file_path, 
        top_n=top_n
    )
    
    # Verify data alignment
    println("\n" * "="^80)
    println("DATA VALIDATION")
    println("="^80)
    
    if length(n_gav) != length(selection_freq)
        error("Length mismatch! n_gav: $(length(n_gav)), selection_freq: $(length(selection_freq))")
    end
    
    println("âœ“ Data lengths match: $(length(n_gav)) individuals")
    println("âœ“ n_gav range: [$(round(minimum(n_gav), digits=2)), $(round(maximum(n_gav), digits=2))]")
    println("âœ“ selection_freq range: [$(round(minimum(selection_freq), digits=3)), $(round(maximum(selection_freq), digits=3))]")
    
    # Quick correlation check
    cor_val = cor(n_gav, selection_freq)
    println("âœ“ Correlation (EBV vs Selection): $(round(cor_val, digits=3))")
    
    if cor_val < 0.3
        println("âš  WARNING: Low correlation detected. Expected positive correlation.")
        println("  This might indicate a data alignment issue.")
    elseif cor_val > 0.5
        println("âœ“ Good correlation detected - data looks aligned!")
    end
    
    # Now run the complete analysis
    println("\n" * "="^80)
    println("RUNNING POLYNOMIAL AND GP REGRESSION ANALYSIS")
    println("="^80)
    
    # Load the main analysis functions
    include("analyze_norway_spruce_data.jl")
    
    # Run analysis
    results = analyze_norway_spruce_data(
        n_gav,
        selection_freq,
        output_file=output_file,
        save_csv=true
    )
    
    # Add MCMC-specific information to results
    results["mcmc_info"] = Dict(
        "n_iterations" => size(mcmc_matrix, 2),
        "top_n_per_iteration" => top_n,
        "mcmc_file" => mcmc_file_path,
        "selection_freq" => selection_freq,
        "individual_ids" => individual_ids
    )
    
    return results
end

# ============================================================================
# QUICK START FUNCTION (For immediate testing)
# ============================================================================

"""
Quick test to verify your data before running full analysis
"""
function quick_test_mcmc_data(n_gav::Vector{Float64}, mcmc_file_path::String; top_n::Int=100)
    println("="^80)
    println("QUICK TEST - Norway Spruce MCMC Data")
    println("="^80)
    
    # Load selection frequencies
    selection_freq, mcmc_matrix, individual_ids = load_mcmc_contributions(
        mcmc_file_path,
        top_n=top_n
    )
    
    # Validate
    println("\n" * "="^80)
    println("VALIDATION CHECK")
    println("="^80)
    
    println("âœ“ n_gav length: $(length(n_gav))")
    println("âœ“ selection_freq length: $(length(selection_freq))")
    
    if length(n_gav) != length(selection_freq)
        println("âŒ ERROR: Lengths don't match!")
        println("  n_gav: $(length(n_gav))")
        println("  selection_freq: $(length(selection_freq))")
        println("\n  Possible solutions:")
        println("  1. Check if n_gav includes all individuals in MCMC matrix")
        println("  2. Check if some individuals were filtered out")
        println("  3. Verify row order matches between files")
        return nothing
    end
    
    println("âœ“ Lengths match!")
    
    # Quick correlation
    cor_val = cor(n_gav, selection_freq)
    println("âœ“ Correlation: $(round(cor_val, digits=3))")
    
    # Quick quadratic fit
    println("\n" * "="^80)
    println("QUICK QUADRATIC FIT TEST")
    println("="^80)
    
    X = hcat(ones(length(n_gav)), n_gav, n_gav.^2)
    Î² = X \\ selection_freq
    y_pred = X * Î²
    
    r2 = 1 - sum((selection_freq .- y_pred).^2) / sum((selection_freq .- mean(selection_freq)).^2)
    
    # Compare to linear
    X_lin = hcat(ones(length(n_gav)), n_gav)
    Î²_lin = X_lin \\ selection_freq
    y_pred_lin = X_lin * Î²_lin
    r2_lin = 1 - sum((selection_freq .- y_pred_lin).^2) / sum((selection_freq .- mean(selection_freq)).^2)
    
    println("Linear RÂ²:    $(round(r2_lin, digits=4))")
    println("Quadratic RÂ²: $(round(r2, digits=4))")
    println("Improvement:  +$(round(r2 - r2_lin, digits=4))")
    println("\nEquation: y = $(round(Î²[1], digits=4)) + $(round(Î²[2], digits=6))x + $(round(Î²[3], digits=8))xÂ²")
    
    if r2 > 0.7
        println("\nâœ“âœ“ EXCELLENT! Data is ready for full analysis.")
        println("\nNext step:")
        println("  results = analyze_norway_spruce_mcmc_data(n_gav, mcmc_file_path)")
    elseif r2 > 0.5
        println("\nâœ“ GOOD! Data looks reasonable.")
        println("\nNext step:")
        println("  results = analyze_norway_spruce_mcmc_data(n_gav, mcmc_file_path)")
    else
        println("\nâš  RÂ² is lower than expected. Check data alignment.")
    end
    
    return selection_freq
end

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
EXAMPLE USAGE:

# You have n_gav already loaded
# Your MCMC file path:
mcmc_file = "C:/Users/JOAH/OneDrive - Skogforsk/Documents/Projekt/Optimum contribution selection/NorwaySpruceData/Save/mcmc_contributions_matrix.csv"

# Option 1: Quick test first (recommended)
include("load_and_analyze_norway_spruce.jl")
selection_freq = quick_test_mcmc_data(n_gav, mcmc_file, top_n=100)

# If test looks good, run full analysis:
results = analyze_norway_spruce_mcmc_data(n_gav, mcmc_file, 
                                         top_n=100,
                                         output_file="my_analysis_results.png")

# Option 2: Run directly (if confident)
results = analyze_norway_spruce_mcmc_data(n_gav, mcmc_file)

# Extract results for paper:
best_model = results["polynomial"][results["best_model"]]
println("Equation: ", best_model[:equation])
println("RÂ² = ", best_model[:r2])
println("F-statistic = ", best_model[:f_statistic])
"""

println("""
================================================================================
NORWAY SPRUCE MCMC ANALYSIS - READY TO USE
================================================================================

You have: n_gav (MAP EBV index)
Your data: mcmc_contributions_matrix.csv (MCMC contributions per iteration)

STEP 1: Set your file path
mcmc_file = "C:/Users/JOAH/OneDrive - Skogforsk/Documents/Projekt/Optimum contribution selection/NorwaySpruceData/Save/mcmc_contributions_matrix.csv"

STEP 2: Run quick test
include("load_and_analyze_norway_spruce.jl")
selection_freq = quick_test_mcmc_data(n_gav, mcmc_file, top_n=100)

STEP 3: If test is good, run full analysis
results = analyze_norway_spruce_mcmc_data(n_gav, mcmc_file, top_n=100)

DONE! You'll get:
- âœ“ Statistical analysis (polynomial + GP)
- âœ“ Publication figure
- âœ“ Manuscript text
- âœ“ Model equation and statistics

================================================================================
""")
