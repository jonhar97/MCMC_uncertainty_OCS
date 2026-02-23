using DataFrames
using CSV
using Plots
using DataStructures
using StatsPlots
using Statistics
using COSMO, JuMP
using DelimitedFiles
using LinearAlgebra
using Distributed
using StatsBase
using Printf
using ProgressMeter
using Dates
using HypothesisTests  # For Kendall correlation

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
THETA = 0.02                          # Genetic constraint parameter
N_SELECTED = 100                      # Number of individuals to select
GAIN_THRESHOLD = 0.2                  # Threshold for filtering low-gain iterations
CONVERGENCE_TOL = 1e-6                # Solver tolerance
MAX_ITER = 10000                      # Maximum solver iterations

# Data paths - UPDATE THESE TO YOUR ACTUAL PATHS
DATA_PATH = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData"
RESULTS_PATH = joinpath(DATA_PATH, "results_JWAS_1218_G_adj_Lev17")
SAVE_PATH = joinpath(DATA_PATH, "Save")
FIGURES_PATH = joinpath(DATA_PATH, "Figures")

# MCMC files
MCMC_FILES = [
    joinpath(RESULTS_PATH, "MCMC_samples_EBV_Hjd17.txt"),
    joinpath(RESULTS_PATH, "MCMC_samples_EBV_Htv17.txt"),
    joinpath(RESULTS_PATH, "MCMC_samples_EBV_Sprant17.txt")
]

# EBV reference files
EBV_FILES = [
    joinpath(RESULTS_PATH, "EBV_Hjd17.txt"),
    joinpath(RESULTS_PATH, "EBV_Htv17.txt"),
    joinpath(RESULTS_PATH, "EBV_Sprant17.txt")
]

G_MATRIX_FILE = joinpath(SAVE_PATH, "JWAS_G_1218_tuned2.txt")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""
    load_mcmc_data(file_paths::Vector{String})

Load MCMC samples from multiple files and return as a vector of DataFrames.
"""
function load_mcmc_data(file_paths::Vector{String})
    println("Loading MCMC data...")
    mcmc_data = DataFrame[]
    
    for (i, file_path) in enumerate(file_paths)
        if !isfile(file_path)
            error("MCMC file not found: $file_path")
        end
        
        data = CSV.read(file_path, DataFrame, delim=',', header=true, missingstrings=["NA"])
        push!(mcmc_data, data)
        println("Loaded MCMC file $i: $(size(data)) samples")
    end
    
    return mcmc_data
end

"""
    load_ebv_data(file_paths::Vector{String})

Load EBV reference data from multiple files.
"""
function load_ebv_data(file_paths::Vector{String})
    println("Loading EBV reference data...")
    ebv_data = DataFrame[]
    
    for (i, file_path) in enumerate(file_paths)
        if !isfile(file_path)
            error("EBV file not found: $file_path")
        end
        
        data = DataFrame(CSV.File(file_path))
        push!(ebv_data, data)
        println("Loaded EBV file $i: $(size(data))")
    end
    
    return ebv_data
end

"""
    load_relationship_matrix(file_path::String)

Load and process the relationship matrix G.
"""
function load_relationship_matrix(file_path::String)
    println("Loading relationship matrix...")
    
    if !isfile(file_path)
        error("Relationship matrix file not found: $file_path")
    end
    
    Amat_t = readdlm(file_path, ',', Float64, '\n', header=false)
    IDs = Amat_t[:, 1]
    G = Amat_t[:, 2:end]
    
    println("Relationship matrix loaded: $(size(G))")
    println("Sample of G matrix:")
    display(G[1:5, 1:5])
    
    return G, IDs
end

# ============================================================================
# CORE ANALYSIS FUNCTIONS
# ============================================================================

"""
    optimum_contribution(A::Matrix, g::Vector, m::Vector, n::Int, theta::Float64)

Solve the optimum contribution selection problem using COSMO optimizer.
Returns the contribution vector and optimization status.
"""
function optimum_contribution(A::Matrix, g::Vector, m::Vector, n::Int, theta::Float64)
    model = Model(optimizer_with_attributes(COSMO.Optimizer, 
                                          "eps_abs" => CONVERGENCE_TOL,
                                          "eps_rel" => CONVERGENCE_TOL,
                                          "max_iter" => MAX_ITER,
                                          "verbose" => false))
    
    @variable(model, x[1:n] >= 0)
    @objective(model, Min, -dot(g, x))
    @constraint(model, sum(m[i] * x[i] for i in 1:n) == 1.0)
    @constraint(model, x' * A * x / 2.0 <= theta)
    
    JuMP.optimize!(model)
    
    status = termination_status(model)
    if status != MOI.OPTIMAL
        @warn "Optimization not optimal: $status"
    end
    
    c = JuMP.value.(x)
    return c, status
end

"""
    calculate_ebv_index(mcmc_data::Vector{DataFrame}, iteration::Int)

Calculate the EBV index for a given MCMC iteration by averaging across traits.
"""
function calculate_ebv_index(mcmc_data::Vector{DataFrame}, iteration::Int)
    n = size(mcmc_data[1], 2)
    ebv_index = Vector{Float64}(undef, n)
    
    for j in 1:n
        ebv_index[j] = mean(mcmc_data[k][iteration, j] for k in 1:length(mcmc_data))
    end
    
    return ebv_index
end

"""
    standardize_ebvs(ebvs::Vector{Float64}, reference_mean::Float64, reference_var::Float64)

Standardize EBVs to have mean 0 and unit variance, then optionally adjust to match reference statistics.
"""
function standardize_ebvs(ebvs::Vector{Float64}; use_reference::Bool=false, 
                         reference_mean::Float64=0.0, reference_var::Float64=1.0)
    ebv_mean = mean(ebvs)
    ebv_var = var(ebvs)
    
    if ebv_var ≈ 0
        @warn "Zero variance in EBVs, returning zeros"
        return zeros(length(ebvs))
    end
    
    centered = ebvs .- ebv_mean
    standardized = centered ./ sqrt(ebv_var)
    
    if use_reference
        return standardized .* sqrt(reference_var) .+ reference_mean
    else
        return standardized
    end
end

"""
    calculate_summary_statistics(contributions::Matrix{Float64})

Calculate various summary statistics for the contribution matrix.
"""
function calculate_summary_statistics(contributions::Matrix{Float64})
    nmc, n = size(contributions)
    
    # Proportion of times each individual is selected (contribution > 0)
    selection_proportions = vec(mean(contributions .> 0, dims=1))
    
    # Mean contribution for each individual
    mean_contributions = vec(mean(contributions, dims=1))
    
    # Coefficient of variation for each individual
    std_contributions = vec(std(contributions, dims=1))
    cv_contributions = std_contributions ./ (mean_contributions .+ 1e-10)  # Avoid division by zero
    
    return (proportions=selection_proportions, 
            means=mean_contributions, 
            stds=std_contributions, 
            cvs=cv_contributions)
end

"""
    calculate_genetic_metrics(contributions::Matrix{Float64}, ebvs::Vector{Float64}, G::Matrix{Float64})

Calculate genetic gain and coancestry for each MCMC iteration.
"""
function calculate_genetic_metrics(contributions::Matrix{Float64}, ebvs::Vector{Float64}, G::Matrix{Float64})
    nmc, n = size(contributions)
    genetic_gains = Vector{Float64}(undef, nmc)
    coancestries = Vector{Float64}(undef, nmc)
    
    for i in 1:nmc
        c = contributions[i, :]
        genetic_gains[i] = dot(c, ebvs)
        coancestries[i] = 0.5 * dot(c, G * c)
    end
    
    return genetic_gains, coancestries
end

"""
    calculate_overlap_analysis(mcmc_contributions::Matrix{Float64}, map_contributions::Vector{Float64}, 
                              n_selected::Int=N_SELECTED)

Perform comprehensive overlap analysis between MCMC and MAP solutions.
"""
function calculate_overlap_analysis(mcmc_contributions::Matrix{Float64}, 
                                  map_contributions::Vector{Float64}, 
                                  n_selected::Int=N_SELECTED)
    nmc, n = size(mcmc_contributions)
    
    # Dynamic threshold for MAP solution
    map_threshold = sort(map_contributions, rev=true)[n_selected]
    map_selected = findall(x -> x >= map_threshold, map_contributions)
    
    overlaps = Vector{Int}(undef, nmc)
    jaccard_similarities = Vector{Float64}(undef, nmc)
    rank_correlations = Vector{Float64}(undef, nmc)
    contribution_correlations = Vector{Float64}(undef, nmc)
    kendall_tau = Vector{Float64}(undef, nmc)
    top_k_precision = Matrix{Float64}(undef, nmc, 5)  # For top 20, 40, 60, 80, 100
    overlap_stability = Vector{Float64}(undef, nmc)
    selection_intensity_ratio = Vector{Float64}(undef, nmc)
    
    for i in 1:nmc
        # Get top n_selected individuals for this iteration
        sorted_indices = sortperm(mcmc_contributions[i, :], rev=true)
        mcmc_selected = sorted_indices[1:n_selected]
        
        # 1. Basic overlap metrics
        overlap_set = intersect(map_selected, mcmc_selected)
        overlaps[i] = length(overlap_set)
        
        # 2. Jaccard similarity
        union_set = union(map_selected, mcmc_selected)
        jaccard_similarities[i] = length(overlap_set) / length(union_set)
        
        # 3. Rank correlation (Spearman)
        rank_correlations[i] = corspearman(map_contributions, mcmc_contributions[i, :])
        
        # 4. Contribution value correlation (Pearson)
        contribution_correlations[i] = cor(map_contributions, mcmc_contributions[i, :])
        
        # 5. Kendall's tau (alternative rank correlation, more robust to outliers)
        kendall_tau[i] = corkendall(map_contributions, mcmc_contributions[i, :])
        
        # 6. Top-k precision for different selection intensities
        k_values = [20, 40, 60, 80, 100]
        for (j, k) in enumerate(k_values)
            if k <= n_selected && k <= n
                map_top_k = sortperm(map_contributions, rev=true)[1:k]
                mcmc_top_k = sorted_indices[1:k]
                overlap_k = length(intersect(map_top_k, mcmc_top_k))
                top_k_precision[i, j] = overlap_k / k
            else
                top_k_precision[i, j] = NaN
            end
        end
        
        # 7. Overlap stability (weighted by contribution values)
        if length(overlap_set) > 0
            overlap_weights = sum(map_contributions[overlap_set]) + sum(mcmc_contributions[i, overlap_set])
            total_weights = sum(map_contributions[map_selected]) + sum(mcmc_contributions[i, mcmc_selected])
            overlap_stability[i] = overlap_weights / total_weights
        else
            overlap_stability[i] = 0.0
        end
        
        # 8. Selection intensity ratio
        map_intensity = sum(map_contributions[map_selected].^2)
        mcmc_intensity = sum(mcmc_contributions[i, mcmc_selected].^2)
        if map_intensity > 0
            selection_intensity_ratio[i] = mcmc_intensity / map_intensity
        else
            selection_intensity_ratio[i] = 1.0
        end
    end
    
    return (overlaps=overlaps, 
            jaccard=jaccard_similarities, 
            rank_corr=rank_correlations,
            contrib_corr=contribution_correlations,
            kendall_tau=kendall_tau,
            top_k_precision=top_k_precision,
            overlap_stability=overlap_stability,
            selection_intensity_ratio=selection_intensity_ratio,
            map_selected=map_selected)
end

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

"""
    create_summary_plots(results::NamedTuple, save_path::String)

Create comprehensive summary plots for the uncertainty analysis.
"""
function create_summary_plots(results::NamedTuple, save_path::String)
    # Plot 1: Genetic gain distribution
    p1 = histogram(results.filtered_gains, bins=30, 
                   title="Genetic Gain Distribution (GC = $THETA)", 
                   xlabel="Rate of genetic progress", 
                   ylabel="Frequency", 
                   label="MCMC BV",
                   alpha=0.7)
    vline!([results.map_gain], label="MAP EBV", lw=3, color=:red)
    
    # Add confidence intervals
    gain_ci = quantile(results.filtered_gains, [0.025, 0.975])
    vline!(gain_ci, label="95% CI", lw=2, color=:gray, linestyle=:dash)
    
    # Plot 2: Overlap distribution
    p2 = histogram(results.overlap_analysis.overlaps, bins=20,
                   title="Selection Overlap with MAP",
                   xlabel="Number of overlapping selections",
                   ylabel="Frequency",
                   label=false,
                   alpha=0.7)
    vline!([mean(results.overlap_analysis.overlaps)], 
           label="Mean overlap", lw=2, color=:red)
    
    # Plot 3: Selection frequency by individual
    sorted_indices = sortperm(results.reference_ebvs)
    p3 = bar(results.summary_stats.proportions[sorted_indices],
             title="Selection Frequency by Individual",
             xlabel="Individual (sorted by EBV)",
             ylabel="Proportion selected",
             label=false,
             alpha=0.7)
    hline!([N_SELECTED/length(results.reference_ebvs)], 
           color=:red, linestyle=:dash, label="Expected frequency")
    
    # Plot 4: Jaccard similarity over iterations
    p4 = plot(results.overlap_analysis.jaccard,
              title="Jaccard Similarity with MAP",
              xlabel="MCMC Iteration",
              ylabel="Jaccard Similarity",
              label=false,
              alpha=0.7)
    hline!([mean(results.overlap_analysis.jaccard)], 
           color=:red, label="Mean similarity")
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
    
    # Save plots
    savefig(combined_plot, joinpath(save_path, "summary_uncertainty_analysis_A_1218.png"))
    savefig(combined_plot, joinpath(save_path, "summary_uncertainty_analysis_A_1218.pdf"))
    
    return combined_plot
end

"""
    create_heatmap_analysis(contributions::Matrix{Float64}, n_selected::Int, save_path::String)

Create heatmap showing selection patterns across MCMC iterations.
"""
function create_heatmap_analysis(contributions::Matrix{Float64}, n_selected::Int, save_path::String)
    nmc, n = size(contributions)
    
    # Create binary selection matrix
    selection_matrix = zeros(Int, nmc, n)
    
    for i in 1:nmc
        sorted_indices = sortperm(contributions[i, :], rev=true)[1:n_selected]
        selection_matrix[i, sorted_indices] .= 1
    end
    
    # Create heatmap
    custom_palette = cgrad([:white, :blue], 2; categorical=true)
    heatmap_plot = heatmap(selection_matrix, 
                          color=custom_palette,
                          title="Selection Patterns Across MCMC Iterations",
                          xlabel="Individual",
                          ylabel="MCMC Iteration",
                          legend=false)
    
    # Save heatmap
    savefig(heatmap_plot, joinpath(save_path, "selection_heatmap_A_1218.png"))
    savefig(heatmap_plot, joinpath(save_path, "selection_heatmap_A_1218.pdf"))
    
end
"""
    create_advanced_overlap_plots(overlap_analysis::NamedTuple, save_path::String)

Create detailed visualizations for all overlap metrics.
"""
function create_advanced_overlap_plots(overlap_analysis::NamedTuple, save_path::String)
    # Plot 1: Correlation comparison
    p1 = scatter(overlap_analysis.rank_corr, overlap_analysis.contrib_corr,
                xlabel="Spearman Rank Correlation",
                ylabel="Pearson Contribution Correlation",
                title="Rank vs. Value Correlations",
                alpha=0.6,
                label=false)
    plot!([0, 1], [0, 1], line=:dash, color=:red, label="Perfect agreement")
    
    # Plot 2: Top-k precision trends
    k_values = [20, 40, 60, 80, 100]
    mean_precisions = [mean(overlap_analysis.top_k_precision[:, i]) for i in 1:5 if !isnan(overlap_analysis.top_k_precision[1, i])]
    valid_k = k_values[1:length(mean_precisions)]
    
    p2 = plot(valid_k, mean_precisions,
              xlabel="Selection Intensity (Top-k)",
              ylabel="Mean Precision",
              title="Precision vs. Selection Intensity",
              marker=:circle,
              linewidth=2,
              label="Mean Precision")
    
    # Add error bars
    std_precisions = [std(overlap_analysis.top_k_precision[:, i]) for i in 1:length(mean_precisions)]
    plot!(valid_k, mean_precisions, ribbon=std_precisions, alpha=0.3, label="±1 SD")
    
    # Plot 3: Stability metrics scatter
    p3 = scatter(overlap_analysis.jaccard, overlap_analysis.overlap_stability,
                xlabel="Jaccard Similarity",
                ylabel="Weighted Overlap Stability",
                title="Selection Stability Comparison",
                alpha=0.6,
                label=false)
    
    # Plot 4: Selection intensity distribution
    p4 = histogram(overlap_analysis.selection_intensity_ratio,
                  xlabel="Selection Intensity Ratio (MCMC/MAP)",
                  ylabel="Frequency",
                  title="Selection Intensity Comparison",
                  alpha=0.7,
                  label=false)
    vline!([1.0], color=:red, linewidth=2, label="Equal intensity")
    vline!([mean(overlap_analysis.selection_intensity_ratio)], 
           color=:blue, linewidth=2, label="Mean ratio")
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
    
    # Save
    savefig(combined_plot, joinpath(save_path, "advanced_overlap_analysis_A_1218.png"))
    savefig(combined_plot, joinpath(save_path, "advanced_overlap_analysis_A_1218.pdf"))
    
    return combined_plot
end

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

"""
    run_uncertainty_analysis()

Main function to run the complete uncertainty impact analysis.
"""
function run_uncertainty_analysis()
"""
    interpret_overlap_metrics(overlap_analysis::NamedTuple)

Provide detailed interpretation of overlap metrics for breeding decisions.
"""
function interpret_overlap_metrics(overlap_analysis::NamedTuple)
    println("\n" * "="^80)
    println("DETAILED OVERLAP METRICS INTERPRETATION")
    println("="^80)
    
    # Basic overlap statistics
    mean_overlap = mean(overlap_analysis.overlaps)
    std_overlap = std(overlap_analysis.overlaps)
    overlap_cv = std_overlap / mean_overlap
    
    println("1. OVERLAP COUNT ANALYSIS:")
    println("   Mean: $(@sprintf("%.1f", mean_overlap))/$N_SELECTED individuals")
    println("   Std: $(@sprintf("%.1f", std_overlap))")
    println("   CV: $(@sprintf("%.3f", overlap_cv))")
    
    if mean_overlap >= 0.8 * N_SELECTED
        println("   → INTERPRETATION: Very stable selection (high confidence)")
    elseif mean_overlap >= 0.6 * N_SELECTED
        println("   → INTERPRETATION: Moderately stable selection")
    else
        println("   → INTERPRETATION: Highly uncertain selection (breeding values matter greatly)")
    end
    
    # Jaccard similarity
    mean_jaccard = mean(overlap_analysis.jaccard)
    println("\n2. JACCARD SIMILARITY:")
    println("   Mean: $(@sprintf("%.3f", mean_jaccard))")
    
    if mean_jaccard >= 0.7
        println("   → INTERPRETATION: High agreement between methods")
    elseif mean_jaccard >= 0.5
        println("   → INTERPRETATION: Moderate agreement")
    else
        println("   → INTERPRETATION: Poor agreement - high uncertainty impact")
    end
    
    # Rank correlations
    mean_spearman = mean(overlap_analysis.rank_corr)
    mean_kendall = mean(overlap_analysis.kendall_tau)
    println("\n3. RANK CORRELATIONS:")
    println("   Spearman ρ: $(@sprintf("%.3f", mean_spearman))")
    println("   Kendall τ: $(@sprintf("%.3f", mean_kendall))")
    
    if mean_spearman >= 0.8
        println("   → INTERPRETATION: Very consistent ranking across all individuals")
    elseif mean_spearman >= 0.6
        println("   → INTERPRETATION: Generally consistent ranking")
    else
        println("   → INTERPRETATION: Inconsistent ranking - uncertainty affects all levels")
    end
    
    # Contribution correlations
    mean_contrib_corr = mean(overlap_analysis.contrib_corr)
    println("\n4. CONTRIBUTION VALUE CORRELATION:")
    println("   Pearson r: $(@sprintf("%.3f", mean_contrib_corr))")
    println("   → INTERPRETATION: How similarly breeding values translate to contributions")
    
    # Top-k precision analysis
    println("\n5. TOP-K PRECISION ANALYSIS:")
    k_values = [20, 40, 60, 80, 100]
    for (i, k) in enumerate(k_values)
        if !isnan(overlap_analysis.top_k_precision[1, i])
            mean_precision_k = mean(overlap_analysis.top_k_precision[:, i])
            println("   Top-$k precision: $(@sprintf("%.3f", mean_precision_k))")
        end
    end
    println("   → INTERPRETATION: Shows if uncertainty affects top vs. marginal selections differently")
    
    # Overlap stability (weighted)
    mean_stability = mean(overlap_analysis.overlap_stability)
    println("\n6. WEIGHTED OVERLAP STABILITY:")
    println("   Mean: $(@sprintf("%.3f", mean_stability))")
    println("   → INTERPRETATION: Accounts for contribution magnitudes, not just selection")
    
    # Selection intensity
    mean_intensity_ratio = mean(overlap_analysis.selection_intensity_ratio)
    println("\n7. SELECTION INTENSITY RATIO:")
    println("   Mean: $(@sprintf("%.3f", mean_intensity_ratio))")
    
    if abs(mean_intensity_ratio - 1.0) < 0.1
        println("   → INTERPRETATION: Similar selection intensity across methods")
    elseif mean_intensity_ratio > 1.1
        println("   → INTERPRETATION: MCMC selections tend to be more concentrated")
    else
        println("   → INTERPRETATION: MAP selections tend to be more concentrated")
    end
    
    # Overall assessment
    println("\n" * "="^50)
    println("OVERALL ASSESSMENT:")
    
    # Create composite uncertainty score
    uncertainty_score = (1 - mean_jaccard) * 0.3 + 
                        (1 - mean_spearman) * 0.3 + 
                        overlap_cv * 0.2 + 
                        abs(mean_intensity_ratio - 1.0) * 0.2
    
    println("Composite uncertainty score: $(@sprintf("%.3f", uncertainty_score)) (0=no uncertainty, 1=maximum uncertainty)")
    
    if uncertainty_score < 0.2
        println("→ BREEDING RECOMMENDATION: Selection is robust to breeding value uncertainty")
        println("  Consider using MAP estimates for practical breeding decisions")
    elseif uncertainty_score < 0.5
        println("→ BREEDING RECOMMENDATION: Moderate uncertainty - consider multiple scenarios")
        println("  Use MCMC-based selection or increase data collection")
    else
        println("→ BREEDING RECOMMENDATION: High uncertainty - breeding value estimation critical")
        println("  Strongly recommend more data collection or conservative selection strategies")
    end
    
    return uncertainty_score
end
    println("OPTIMUM CONTRIBUTION SELECTION - UNCERTAINTY IMPACT ANALYSIS")
    println("="^80)
    
    # Load all data
    mcmc_data = load_mcmc_data(MCMC_FILES)
    ebv_data = load_ebv_data(EBV_FILES)
    G, IDs = load_relationship_matrix(G_MATRIX_FILE)
    
    # Get dimensions
    nmc, n = size(mcmc_data[1])
    m = ones(n)  # Equal mating opportunities
    
    println("Analysis parameters:")
    println("  - Number of MCMC iterations: $nmc")
    println("  - Number of individuals: $n")
    println("  - Genetic constraint (θ): $THETA")
    println("  - Number to select: $N_SELECTED")
    
    # Calculate reference EBVs (MAP estimates)
    println("\nCalculating reference EBVs...")
    reference_ebvs = Vector{Float64}(undef, n)
    for i in 1:n
        reference_ebvs[i] = mean(ebv_data[k][i, 2] for k in 1:length(ebv_data))
    end
    
    # Standardize reference EBVs
    ref_mean = mean(reference_ebvs)
    ref_var = var(reference_ebvs)
    standardized_ref_ebvs = standardize_ebvs(reference_ebvs)
    
    # Calculate MAP contribution
    println("Calculating MAP contribution...")
    c_map, map_status = optimum_contribution(G, standardized_ref_ebvs, m, n, THETA)
    map_gain = dot(c_map, standardized_ref_ebvs)
    map_coancestry = 0.5 * dot(c_map, G * c_map)
    
    println("MAP Results:")
    println("  - Genetic gain: $(@sprintf("%.6f", map_gain))")
    println("  - Coancestry: $(@sprintf("%.6f", map_coancestry))")
    println("  - Optimization status: $map_status")
    
    # Pre-allocate arrays for MCMC analysis
    println("\nRunning MCMC uncertainty analysis...")
    contributions = Matrix{Float64}(undef, nmc, n)
    gains = Vector{Float64}(undef, nmc)
    coancestries = Vector{Float64}(undef, nmc)
    optimization_statuses = Vector{MOI.TerminationStatusCode}(undef, nmc)
    
    # Temporary arrays for efficiency
    ebv_temp = Vector{Float64}(undef, n)
    
    # Main MCMC loop with progress reporting
    println("Processing MCMC iterations...")
    failed_optimizations = 0
    
    for i in 1:nmc
        if i % 100 == 0 || i == nmc
            println("  Processed $i/$nmc iterations")
        end
        
        # Calculate EBV index for this iteration
        ebv_index = calculate_ebv_index(mcmc_data, i)
        
        # Standardize EBVs
        standardized_ebvs = standardize_ebvs(ebv_index)
        
        # Solve OCS problem
        c, status = optimum_contribution(G, standardized_ebvs, m, n, THETA)
        
        # Store results
        contributions[i, :] = c
        gains[i] = dot(c, standardized_ebvs)
        coancestries[i] = 0.5 * dot(c, G * c)
        optimization_statuses[i] = status
        
        if status != MOI.OPTIMAL
            failed_optimizations += 1
        end
    end
    
    if failed_optimizations > 0
        @warn "Failed optimizations: $failed_optimizations/$nmc"
    end
    
    # Filter out low-gain iterations
    println("\nFiltering results...")
    valid_indices = gains .>= GAIN_THRESHOLD
    n_filtered = sum(valid_indices)
    
    println("  - Valid iterations after filtering: $n_filtered/$nmc")
    println("  - Filtering threshold: $GAIN_THRESHOLD")
    
    filtered_contributions = contributions[valid_indices, :]
    filtered_gains = gains[valid_indices]
    filtered_coancestries = coancestries[valid_indices]
    
    # Calculate summary statistics
    println("Calculating summary statistics...")
    summary_stats = calculate_summary_statistics(filtered_contributions)
    
    # Perform overlap analysis
    println("Performing overlap analysis...")
    overlap_analysis = calculate_overlap_analysis(filtered_contributions, c_map, N_SELECTED)
    
    # Interpret overlap metrics
    uncertainty_score = interpret_overlap_metrics(overlap_analysis)
    
    # Calculate confidence intervals
    gain_ci = quantile(filtered_gains, [0.025, 0.975])
    coancestry_ci = quantile(filtered_coancestries, [0.025, 0.975])
    overlap_mean = mean(overlap_analysis.overlaps)
    jaccard_mean = mean(overlap_analysis.jaccard)
    
    # Print summary results
    println("\n" * "="^80)
    println("SUMMARY RESULTS")
    println("="^80)
    println("Genetic Gain:")
    println("  - MAP estimate: $(@sprintf("%.6f", map_gain))")
    println("  - MCMC mean: $(@sprintf("%.6f", mean(filtered_gains)))")
    println("  - MCMC 95% CI: [$(@sprintf("%.6f", gain_ci[1])), $(@sprintf("%.6f", gain_ci[2]))]")
    println("  - MCMC std: $(@sprintf("%.6f", std(filtered_gains)))")
    println()
    println("Coancestry:")
    println("  - MAP estimate: $(@sprintf("%.6f", map_coancestry))")
    println("  - MCMC mean: $(@sprintf("%.6f", mean(filtered_coancestries)))")
    println("  - MCMC 95% CI: [$(@sprintf("%.6f", coancestry_ci[1])), $(@sprintf("%.6f", coancestry_ci[2]))]")
    println()
    println("Selection Overlap with MAP:")
    println("  - Mean overlap: $(@sprintf("%.1f", overlap_mean))/$N_SELECTED individuals")
    println("  - Mean Jaccard similarity: $(@sprintf("%.4f", jaccard_mean))")
    println("  - Overlap range: [$(minimum(overlap_analysis.overlaps)), $(maximum(overlap_analysis.overlaps))]")
    
    # Create comprehensive results structure
    results = (
        # Data
        mcmc_contributions = filtered_contributions,
        map_contribution = c_map,
        reference_ebvs = standardized_ref_ebvs,
        
        # Metrics
        mcmc_gains = filtered_gains,
        map_gain = map_gain,
        filtered_gains = filtered_gains,
        mcmc_coancestries = filtered_coancestries,
        
        # Analysis results
        summary_stats = summary_stats,
        overlap_analysis = overlap_analysis,
        
        # Metadata
        n_iterations = n_filtered,
        n_individuals = n,
        failed_optimizations = failed_optimizations
    )
    
    # Create visualizations
    println("\nCreating visualizations...")
    mkpath(FIGURES_PATH)  # Ensure directory exists
    
    summary_plot = create_summary_plots(results, FIGURES_PATH)
    selection_matrix, heatmap_plot = create_heatmap_analysis(filtered_contributions, N_SELECTED, FIGURES_PATH)
    advanced_overlap_plot = create_advanced_overlap_plots(overlap_analysis, FIGURES_PATH)
    
    # Save results to files
    println("Saving results...")
    mkpath(SAVE_PATH)  # Ensure directory exists
    
    try
        # Save contribution matrix
        println("  Saving contribution matrix...")
        writedlm(joinpath(SAVE_PATH, "mcmc_contributions_matrix_A_1218.csv"), filtered_contributions, ',')
        
        # Save summary statistics
        println("  Saving summary statistics...")
        summary_data = hcat(summary_stats.proportions, standardized_ref_ebvs, summary_stats.means, summary_stats.stds)
        summary_headers = ["Selection_Proportion", "Reference_EBV", "Mean_Contribution", "Std_Contribution"]
        println("    Summary data dimensions: $(size(summary_data))")
        println("    Summary headers length: $(length(summary_headers))")
        summary_data_with_headers = vcat(reshape(summary_headers, 1, :), summary_data)
        writedlm(joinpath(SAVE_PATH, "summary_statistics_A_1218.csv"), summary_data_with_headers, ',')
        
        # Save overlap analysis
        println("  Saving overlap analysis...")
        overlap_data = hcat(overlap_analysis.overlaps, overlap_analysis.jaccard, overlap_analysis.rank_corr,
                           overlap_analysis.contrib_corr, overlap_analysis.kendall_tau, 
                           overlap_analysis.overlap_stability, overlap_analysis.selection_intensity_ratio)
        overlap_headers = ["Overlap_Count", "Jaccard_Similarity", "Rank_Correlation", "Contribution_Correlation", 
                          "Kendall_Tau", "Overlap_Stability", "Selection_Intensity_Ratio"]
        println("    Overlap data dimensions: $(size(overlap_data))")
        println("    Overlap headers length: $(length(overlap_headers))")
        overlap_data_with_headers = vcat(reshape(overlap_headers, 1, :), overlap_data)
        writedlm(joinpath(SAVE_PATH, "overlap_analysis_A_1218.csv"), overlap_data_with_headers, ',')
        
        # Save top-k precision analysis
        println("  Saving top-k precision analysis...")
        topk_headers = ["Top20", "Top40", "Top60", "Top80", "Top100"]
        println("    Top-k data dimensions: $(size(overlap_analysis.top_k_precision))")
        println("    Top-k headers length: $(length(topk_headers))")
        topk_data_with_headers = vcat(reshape(topk_headers, 1, :), overlap_analysis.top_k_precision)
        writedlm(joinpath(SAVE_PATH, "top_k_precision_A_1218.csv"), topk_data_with_headers, ',')
        
        println("  All files saved successfully!")
        
    catch e
        println("Error saving files: $e")
        println("Attempting to save with alternative method...")
        
        # Alternative saving method without headers
        writedlm(joinpath(SAVE_PATH, "mcmc_contributions_matrix_backup_A_1218.csv"), filtered_contributions, ',')
        
        summary_data = hcat(summary_stats.proportions, standardized_ref_ebvs, summary_stats.means, summary_stats.stds)
        writedlm(joinpath(SAVE_PATH, "summary_statistics_backup_A_1218.csv"), summary_data, ',')
        
        overlap_data = hcat(overlap_analysis.overlaps, overlap_analysis.jaccard, overlap_analysis.rank_corr,
                           overlap_analysis.contrib_corr, overlap_analysis.kendall_tau, 
                           overlap_analysis.overlap_stability, overlap_analysis.selection_intensity_ratio)
        writedlm(joinpath(SAVE_PATH, "overlap_analysis_backup_A_1218.csv"), overlap_data, ',')
        
        writedlm(joinpath(SAVE_PATH, "top_k_precision_backup_A_1218.csv"), overlap_analysis.top_k_precision, ',')
        
        println("  Backup files saved without headers!")
    end
    
    println("\nAnalysis completed successfully!")
    println("Results saved to: $SAVE_PATH")
    println("Figures saved to: $FIGURES_PATH")
    
    return results
end

# ============================================================================
# EXECUTION
# ============================================================================

# Run the analysis
println("Starting uncertainty impact analysis for optimum contribution selection...")
results = run_uncertainty_analysis()

println("\nAnalysis completed! Check the results in:")
println("- Results: $SAVE_PATH")
println("- Figures: $FIGURES_PATH")