"""
Optimized MCMC Robustness Analysis for Norway Spruce
====================================================

Two-Step Candidate Selection Strategy:
1. Characterize all MAP-OCS selected individuals (~160)
2. Add 150 high-EBV unselected individuals as replacement pool

This eliminates redundancy and focuses computational effort where it matters.

Author: Jon Ahlinder (Skogforsk)
Date: February 2026
"""

# Prevent duplicate loading warning
if !@isdefined(NORWAY_SPRUCE_MCMC_LOADED)
    global NORWAY_SPRUCE_MCMC_LOADED = true
    println("Loading Norway Spruce MCMC Robustness Analysis...")
else
    println("Reloading Norway Spruce MCMC Robustness Analysis...")
    println("  (Functions will be redefined)")
end

using DataFrames, CSV, Statistics, LinearAlgebra
using COSMO, JuMP, Plots, StatsPlots
using DelimitedFiles, Measures, JLD2, FileIO
using HypothesisTests, Distributions

# ============================================================================
# CONFIGURATION
# ============================================================================

# Analysis parameters (can be modified between runs)
THETA = 0.02  # Coancestry constraint
SELECTION_THRESHOLD = 1e-6  # Contribution threshold for "selected"
N_REPLACEMENT_CANDIDATES = 150  # High-EBV unselected individuals
HIGH_RISK_PERCENTILE = 0.25  # Bottom 25% of selected individuals

# Selection index formula: (HJD17 + HTV17 - SPRANT17) / 3
# Where:
#   HJD17 = Height Dominant (positive weight - want MORE height)
#   HTV17 = Height Mean (positive weight - want MORE height)
#   SPRANT17 = Spiral Grain (negative weight - want LESS spiral grain)
# To modify index, change formula in:
#   - load_and_prepare_data() for MAP calculation
#   - load_mcmc_breeding_values() for MCMC samples

# File paths (can be modified for different systems)
DATA_PATH = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData"
SAVE_PATH = joinpath(DATA_PATH, "Save")
FIGURES_PATH = joinpath(DATA_PATH, "Figures")
RESULTS_PATH = joinpath(DATA_PATH, "results_JWAS_1218_G_adj_Lev17")

# ============================================================================
# CORE OCS FUNCTION
# ============================================================================

"""
    OC(A, g, m, n, theta) -> Vector{Float64}

Optimum contribution selection with coancestry constraint.
"""
function OC(A::Matrix{Float64}, g::Vector{Float64}, m::Vector{Float64}, 
            n::Int, theta::Float64)
    model = Model(COSMO.Optimizer)
    set_silent(model)  # Suppress solver output
    
    @variable(model, x[1:n] >= 0)
    @objective(model, Max, dot(g, x))
    @constraint(model, sum(m[i]*x[i] for i in 1:n) == 1.0)
    @constraint(model, x'*A*x/2.0 <= theta)
    
    optimize!(model)
    return value.(x)
end

"""
    OC_constrained(A, g, m, n, theta, exclude_indices) -> (c, g, coancestry, success)

OCS with individuals constrained to zero contribution.
"""
function OC_constrained(A::Matrix{Float64}, g::Vector{Float64}, m::Vector{Float64},
                       n::Int, theta::Float64, exclude_indices::Vector{Int})
    model = Model(COSMO.Optimizer)
    set_silent(model)
    
    @variable(model, x[1:n] >= 0)
    @objective(model, Max, dot(g, x))
    @constraint(model, sum(m[i]*x[i] for i in 1:n) == 1.0)
    @constraint(model, x'*A*x/2.0 <= theta)
    
    # Force excluded individuals to zero
    for idx in exclude_indices
        @constraint(model, x[idx] == 0.0)
    end
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        c = value.(x)
        gain = c' * g
        coancestry = 0.5 * c' * A * c
        return c, gain, coancestry, true
    else
        return zeros(n), 0.0, 0.0, false
    end
end

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

"""
    load_mcmc_breeding_values(use_cache=true) -> Matrix{Float64}

Load MCMC samples for three traits and calculate selection index.
Selection index: (Hjd17 + Htv17 - Sprant17) / 3

If use_cache=true, will save/load from cached JLD2 file for faster subsequent runs.

Returns: Matrix of size (n_mcmc_samples × n_individuals)
"""
function load_mcmc_breeding_values(use_cache::Bool=true)
    cache_file = joinpath(SAVE_PATH, "mcmc_index_cache.jld2")
    
    # Try to load from cache first
    if use_cache && isfile(cache_file)
        println("\n  Loading MCMC index from cache...")
        try
            @load cache_file mcmc_index
            println("    ✓ Loaded from cache: $(size(mcmc_index, 1)) samples × " *
                    "$(size(mcmc_index, 2)) individuals")
            return mcmc_index
        catch e
            println("    ⚠ Cache load failed: $e")
            println("    → Recalculating from source files...")
        end
    end
    
    println("\n  Loading three-trait MCMC samples...")
    
    # Define file paths using RESULTS_PATH
    mcmc_file1 = joinpath(RESULTS_PATH, "MCMC_samples_EBV_Hjd17.txt")
    mcmc_file2 = joinpath(RESULTS_PATH, "MCMC_samples_EBV_Htv17.txt")
    mcmc_file3 = joinpath(RESULTS_PATH, "MCMC_samples_EBV_Sprant17.txt")
    
    # Check if files exist
    for (i, file) in enumerate([mcmc_file1, mcmc_file2, mcmc_file3])
        if !isfile(file)
            error("MCMC file not found: $file")
        end
    end
    
    # Load each trait
    println("    - Loading Hjd17 (Height Dominant)...")
    mcmc1 = CSV.read(mcmc_file1, DataFrame, delim=',', header=true, 
                    missingstrings=["NA"])
    
    println("    - Loading Htv17 (Height Mean)...")
    mcmc2 = CSV.read(mcmc_file2, DataFrame, delim=',', header=true, 
                    missingstrings=["NA"])
    
    println("    - Loading Sprant17 (Spiral Grain)...")
    mcmc3 = CSV.read(mcmc_file3, DataFrame, delim=',', header=true, 
                    missingstrings=["NA"])
    
    # Convert to matrices (remove any ID/iteration columns if present)
    # Check if first column is non-numeric (IDs) or if it's just sequential numbers
    function extract_ebv_matrix(df::DataFrame)
        # Try to determine if first column is IDs or data
        first_col = df[:, 1]
        
        # If first column has non-numeric or is called "ID", "iteration", etc.
        col_name = String(names(df)[1])
        if occursin(r"^(ID|id|iter|Iter|iteration)"i, col_name)
            println("      → Detected ID/iteration column: '$col_name', skipping")
            return Matrix{Float64}(df[:, 2:end])
        end
        
        # If first column looks like sequential integers 1,2,3...
        # (likely iteration numbers), skip it
        if eltype(first_col) <: Integer
            if all(first_col .== 1:length(first_col))
                println("      → Detected sequential iteration column, skipping")
                return Matrix{Float64}(df[:, 2:end])
            end
        end
        
        # Otherwise, assume all columns are EBV data
        println("      → Using all columns as EBV data")
        return Matrix{Float64}(df)
    end
    
    mat1 = extract_ebv_matrix(mcmc1)
    mat2 = extract_ebv_matrix(mcmc2)
    mat3 = extract_ebv_matrix(mcmc3)
    
    # Verify dimensions match
    if size(mat1) != size(mat2) || size(mat1) != size(mat3)
        error("MCMC sample dimensions don't match: " *
              "$(size(mat1)) vs $(size(mat2)) vs $(size(mat3))")
    end
    
    n_samples, n_individuals = size(mat1)
    println("    ✓ Dimensions: $n_samples MCMC samples × $n_individuals individuals")
    
    # Calculate selection index: (Hjd17 + Htv17 - Sprant17) / 3
    println("    - Calculating selection index: (Hjd17 + Htv17 - Sprant17) / 3")
    mcmc_index = (mat1 .+ mat2 .- mat3) ./ 3.0
    
    println("    ✓ Selection index calculated")
    println("    ✓ Range: [$(round(minimum(mcmc_index), digits=2)), " *
            "$(round(maximum(mcmc_index), digits=2))]")
    
    # Save to cache for future use
    if use_cache
        try
            @save cache_file mcmc_index
            println("    ✓ Saved to cache: $cache_file")
        catch e
            println("    ⚠ Cache save failed: $e")
        end
    end
    
    return mcmc_index
end

"""
    load_and_prepare_data() -> (G, IDs, n_gav, m, n, mcmc_bv, c_map)

Load G-matrix, breeding values, and MCMC samples.
"""
function load_and_prepare_data()
    println("="^80)
    println("LOADING DATA")
    println("="^80)
    
    # Load G-matrix
    println("\n1. Loading G-matrix...")
    Amat_t = readdlm(joinpath(SAVE_PATH, "Gmat_1218_spruce_PDF.txt"), 
                     ',', Float64, '\n', header=false)
    IDs = Amat_t[:, 1]
    G = Amat_t[:, 2:end]
    n = size(G, 1)
    println("   ✓ Loaded: $n individuals")
    
    # Load MAP breeding values
    println("\n2. Loading MAP breeding values...")
    
    # EBV files for MAP calculation
    EBV_FILES = Dict(
        "HJD17" => joinpath(RESULTS_PATH, "EBV_Hjd17.txt"),
        "HTV17" => joinpath(RESULTS_PATH, "EBV_Htv17.txt"),
        "SPRANT17" => joinpath(RESULTS_PATH, "EBV_Sprant17.txt")
    )
    
    println("   - Loading Hjd17 (Height Dominant)...")
    hjd17 = CSV.read(EBV_FILES["HJD17"], DataFrame, delim=',', header=true, 
                    missingstring="NA")
    
    println("   - Loading Htv17 (Height Mean)...")
    htv17 = CSV.read(EBV_FILES["HTV17"], DataFrame, delim=',', header=true, 
                    missingstring="NA")
    
    println("   - Loading Sprant17 (Spiral Grain)...")
    sprant17 = CSV.read(EBV_FILES["SPRANT17"], DataFrame, delim=',', header=true, 
                       missingstring="NA")
    
    println("   ✓ EBV files loaded")
    
    # Calculate aggregate index: (HJD17 + HTV17 - SPRANT17) / 3
    println("   - Calculating MAP index: (HJD17 + HTV17 - SPRANT17) / 3")
    gav = (htv17[!, 2] .+ hjd17[!, 2] .- sprant17[!, 2]) ./ 3.0
    
    # Standardize
    n_gav = (gav .- mean(gav)) ./ std(gav)
    println("   ✓ MAP index calculated and standardized")
    
    # Load MCMC breeding values
    println("\n3. Loading MCMC breeding values...")
    mcmc_breeding_values = load_mcmc_breeding_values()
    
    n_mcmc = size(mcmc_breeding_values, 1)
    println("   ✓ MCMC samples: $n_mcmc × $(size(mcmc_breeding_values, 2)) individuals")
    
    # Sex constraint (assuming equal for all)
    m = ones(Float64, n)
    
    # Run MAP-OCS
    println("\n4. Running MAP-OCS...")
    c_map = OC(G, n_gav, m, n, THETA)
    n_selected = sum(c_map .> SELECTION_THRESHOLD)
    gain_map = c_map' * n_gav
    coancestry_map = 0.5 * c_map' * G * c_map
    
    println("   ✓ MAP-OCS complete")
    println("     Selected: $n_selected individuals")
    println("     Genetic gain: $(round(gain_map, digits=6))")
    println("     Coancestry: $(round(coancestry_map, digits=6))")
    
    return G, IDs, n_gav, m, n, mcmc_breeding_values, c_map
end

# ============================================================================
# TWO-STEP CANDIDATE SELECTION
# ============================================================================

"""
    select_robustness_candidates(c_map, n_gav, n_replacement) -> Vector{Int}

Two-step candidate selection:
1. All MAP-OCS selected individuals
2. Top N unselected individuals by breeding value
"""
function select_robustness_candidates(c_map::Vector{Float64}, n_gav::Vector{Float64},
                                     n_replacement::Int=N_REPLACEMENT_CANDIDATES)
    println("\n" * "="^80)
    println("TWO-STEP CANDIDATE SELECTION")
    println("="^80)
    
    # Step 1: All selected individuals
    selected_indices = findall(c_map .> SELECTION_THRESHOLD)
    n_selected = length(selected_indices)
    println("\nStep 1: MAP-OCS selected individuals")
    println("  Count: $n_selected")
    
    # Step 2: Top unselected by EBV
    unselected_indices = findall(c_map .<= SELECTION_THRESHOLD)
    sorted_unselected = unselected_indices[sortperm(n_gav[unselected_indices], rev=true)]
    top_unselected = sorted_unselected[1:min(n_replacement, length(sorted_unselected))]
    
    println("\nStep 2: High-EBV replacement candidates")
    println("  Count: $(length(top_unselected))")
    println("  EBV range: [$(round(minimum(n_gav[top_unselected]), digits=2)), " *
            "$(round(maximum(n_gav[top_unselected]), digits=2))]")
    
    # Combine
    all_candidates = vcat(selected_indices, top_unselected)
    
    println("\nTotal candidates for robustness analysis: $(length(all_candidates))")
    println("  Selected: $n_selected ($(round(n_selected/length(all_candidates)*100, digits=1))%)")
    println("  Replacements: $(length(top_unselected)) " *
            "($(round(length(top_unselected)/length(all_candidates)*100, digits=1))%)")
    
    return all_candidates
end

# ============================================================================
# ROBUSTNESS ANALYSIS
# ============================================================================

"""
    calculate_robustness_scores(mcmc_bv, G, m, n, theta, candidates, c_map) 
        -> DataFrame

Calculate robustness scores for candidate individuals using MCMC sampling.
"""
function calculate_robustness_scores(mcmc_breeding_values::Matrix{Float64},
                                    G::Matrix{Float64}, m::Vector{Float64},
                                    n::Int, theta::Float64,
                                    candidate_indices::Vector{Int},
                                    c_map::Vector{Float64})
    println("\n" * "="^80)
    println("ROBUSTNESS ANALYSIS")
    println("="^80)
    
    n_candidates = length(candidate_indices)
    n_mcmc_total = size(mcmc_breeding_values, 1)
    
    # Sample MCMC iterations for efficiency
    n_samples = min(100, n_mcmc_total)
    sample_indices = rand(1:n_mcmc_total, n_samples)
    
    println("\nAnalyzing $n_candidates candidates")
    println("Using $n_samples MCMC samples (of $n_mcmc_total available)")
    
    # Initialize arrays
    robustness_scores = zeros(Float64, n_candidates)
    baseline_gains = Float64[]
    
    # Progress tracking
    progress_interval = max(1, n_samples ÷ 10)
    
    for (iter_count, mcmc_idx) in enumerate(sample_indices)
        if iter_count % progress_interval == 0
            println("  Progress: $iter_count / $n_samples")
        end
        
        # Standardize this MCMC iteration's breeding values
        iteration_bvs = mcmc_breeding_values[mcmc_idx, :]
        n_iter = (iteration_bvs .- mean(iteration_bvs)) ./ std(iteration_bvs)
        
        # Baseline OCS for this iteration
        c_iter_base = OC(G, n_iter, m, n, theta)
        g_iter_base = c_iter_base' * n_iter
        push!(baseline_gains, g_iter_base)
        
        # Test each candidate's impact
        for (cand_idx, individual) in enumerate(candidate_indices)
            # Force individual to be worst
            modified_bvs = copy(n_iter)
            modified_bvs[individual] = minimum(n_iter) - 1.0
            
            # OCS with modified breeding values
            c_modified = OC(G, modified_bvs, m, n, theta)
            g_modified = c_modified' * n_iter
            
            # Accumulate impact
            robustness_scores[cand_idx] += (g_iter_base - g_modified)
        end
    end
    
    # Normalize by number of samples
    robustness_scores ./= n_samples
    
    # Calculate metrics
    mean_baseline = mean(baseline_gains)
    percentage_impact = (robustness_scores ./ mean_baseline) * 100
    standardized = (robustness_scores .- mean(robustness_scores)) ./ std(robustness_scores)
    
    # Create results DataFrame
    results = DataFrame(
        individual = candidate_indices,
        selected_in_map = [c_map[ind] > SELECTION_THRESHOLD for ind in candidate_indices],
        map_contribution = c_map[candidate_indices],
        breeding_value = (nothing),  # Will be filled externally
        robustness_score = robustness_scores,
        percentage_impact = percentage_impact,
        standardized_robustness = standardized
    )
    
    sort!(results, :robustness_score, rev=true)
    
    println("\n✓ Robustness analysis complete")
    println("  Robustness score range: [$(round(minimum(robustness_scores), digits=6)), " *
            "$(round(maximum(robustness_scores), digits=6))]")
    println("  Percentage impact range: [$(round(minimum(percentage_impact), digits=2))%, " *
            "$(round(maximum(percentage_impact), digits=2))%]")
    
    return results
end

# ============================================================================
# RISK CLASSIFICATION
# ============================================================================

"""
    classify_high_risk_individuals(robustness_results, c_map, percentile) 
        -> Vector{Int}

Identify high-risk individuals: bottom percentile of MAP-selected by robustness.
"""
function classify_high_risk_individuals(robustness_results::DataFrame,
                                        c_map::Vector{Float64},
                                        percentile::Float64=HIGH_RISK_PERCENTILE)
    println("\n" * "="^80)
    println("RISK CLASSIFICATION")
    println("="^80)
    
    # Focus on MAP-selected individuals only
    selected_results = filter(row -> row.selected_in_map, robustness_results)
    n_selected = nrow(selected_results)
    
    println("\nClassifying MAP-selected individuals:")
    println("  Total selected: $n_selected")
    
    # Calculate exclusion threshold
    n_exclude = Int(ceil(n_selected * percentile))
    sorted_by_robustness = sort(selected_results, :robustness_score)
    high_risk = sorted_by_robustness[1:n_exclude, :individual]
    
    threshold_value = sorted_by_robustness[n_exclude, :robustness_score]
    
    println("  Bottom $(percentile*100)% threshold: $n_exclude individuals")
    println("  Robustness threshold: $(round(threshold_value, digits=6))")
    println("  Percentage impact range: " *
            "[$(round(minimum(sorted_by_robustness[1:n_exclude, :percentage_impact]), digits=2))%, " *
            "$(round(maximum(sorted_by_robustness[1:n_exclude, :percentage_impact]), digits=2))%]")
    
    return high_risk
end

# ============================================================================
# VARIANCE ANALYSIS
# ============================================================================

"""
    calculate_variance_metrics(mcmc_bv, c_map, c_constrained, G) 
        -> (metrics_df, data_df)

Evaluate both solutions across all MCMC iterations.
"""
function calculate_variance_metrics(mcmc_breeding_values::Matrix{Float64},
                                   c_map::Vector{Float64},
                                   c_constrained::Vector{Float64},
                                   G::Matrix{Float64})
    println("\n" * "="^80)
    println("VARIANCE COMPARISON")
    println("="^80)
    
    n_samples = size(mcmc_breeding_values, 1)
    
    # Initialize storage
    gains_map = zeros(Float64, n_samples)
    gains_const = zeros(Float64, n_samples)
    coanc_map = zeros(Float64, n_samples)
    coanc_const = zeros(Float64, n_samples)
    
    # Progress
    progress_interval = max(1, n_samples ÷ 20)
    
    println("\nEvaluating across $n_samples MCMC iterations...")
    
    for i in 1:n_samples
        if i % progress_interval == 0
            println("  Progress: $i / $n_samples")
        end
        
        # Standardize
        iteration_bvs = mcmc_breeding_values[i, :]
        n_iter = (iteration_bvs .- mean(iteration_bvs)) ./ std(iteration_bvs)
        
        # Calculate gains
        gains_map[i] = c_map' * n_iter
        gains_const[i] = c_constrained' * n_iter
        
        # Calculate coancestries
        coanc_map[i] = 0.5 * c_map' * G * c_map
        coanc_const[i] = 0.5 * c_constrained' * G * c_constrained
    end
    
    # Summary metrics
    metrics = DataFrame(
        solution = ["MAP-OCS", "Constrained"],
        mean_gain = [mean(gains_map), mean(gains_const)],
        std_gain = [std(gains_map), std(gains_const)],
        cv_gain = [std(gains_map)/abs(mean(gains_map)), 
                  std(gains_const)/abs(mean(gains_const))],
        mean_coancestry = [mean(coanc_map), mean(coanc_const)],
        std_coancestry = [std(coanc_map), std(coanc_const)]
    )
    
    # Full data for plotting
    data = DataFrame(
        iteration = repeat(1:n_samples, 2),
        solution = vcat(fill("MAP-OCS", n_samples), fill("Constrained", n_samples)),
        genetic_gain = vcat(gains_map, gains_const),
        coancestry = vcat(coanc_map, coanc_const)
    )
    
    println("\n✓ Variance analysis complete")
    println("\nMAP-OCS:")
    println("  Mean gain: $(round(metrics[1, :mean_gain], digits=6)) ± " *
            "$(round(metrics[1, :std_gain], digits=6))")
    println("  CV: $(round(metrics[1, :cv_gain], digits=4))")
    
    println("\nConstrained:")
    println("  Mean gain: $(round(metrics[2, :mean_gain], digits=6)) ± " *
            "$(round(metrics[2, :std_gain], digits=6))")
    println("  CV: $(round(metrics[2, :cv_gain], digits=4))")
    
    gain_loss = metrics[2, :mean_gain] - metrics[1, :mean_gain]
    cv_improvement = metrics[1, :cv_gain] - metrics[2, :cv_gain]
    
    println("\nTrade-off:")
    println("  Genetic gain change: $(round(gain_loss, digits=6))")
    println("  CV improvement: $(round(cv_improvement, digits=6))")
    
    return metrics, data
end

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

"""
    perform_statistical_tests(variance_data) -> DataFrame

Comprehensive hypothesis testing of solution differences.
"""
function perform_statistical_tests(variance_data::DataFrame)
    println("\n" * "="^80)
    println("STATISTICAL HYPOTHESIS TESTING")
    println("="^80)
    
    map_gains = filter(row -> row.solution == "MAP-OCS", variance_data).genetic_gain
    const_gains = filter(row -> row.solution == "Constrained", variance_data).genetic_gain
    
    # 1. t-test for mean difference
    ttest = UnequalVarianceTTest(const_gains, map_gains)
    
    # 2. F-test for variance equality
    ftest = VarianceFTest(map_gains, const_gains)
    
    # 3. Kolmogorov-Smirnov test for distribution difference
    kstest = ApproximateTwoSampleKSTest(map_gains, const_gains)
    
    # 4. Mann-Whitney U test (non-parametric)
    mwtest = MannWhitneyUTest(map_gains, const_gains)
    
    # 5. Cohen's d effect size
    pooled_std = sqrt((var(map_gains) + var(const_gains)) / 2)
    cohens_d = (mean(const_gains) - mean(map_gains)) / pooled_std
    
    results = DataFrame(
        Test = ["t-test (mean)", "F-test (variance)", "KS-test (distribution)", 
                "Mann-Whitney U", "Cohen's d"],
        Statistic = [ttest.t, ftest.F, kstest.δ, mwtest.U, cohens_d],
        P_Value = [pvalue(ttest), pvalue(ftest), pvalue(kstest), pvalue(mwtest), NaN],
        Significant_05 = [pvalue(ttest) < 0.05, pvalue(ftest) < 0.05,
                         pvalue(kstest) < 0.05, pvalue(mwtest) < 0.05, abs(cohens_d) > 0.2]
    )
    
    println("\nTest Results:")
    for row in eachrow(results)
        sig = row.Significant_05 ? " ***" : ""
        if !isnan(row.P_Value)
            println("  $(row.Test): p = $(round(row.P_Value, digits=6))$sig")
        else
            println("  $(row.Test): d = $(round(row.Statistic, digits=4))")
        end
    end
    
    return results
end

# ============================================================================
# VISUALIZATION
# ============================================================================

"""
    create_variance_comparison_plot(variance_data, metrics) -> Plot

Four-panel comparison plot.
"""
function create_variance_comparison_plot(variance_data::DataFrame,
                                        metrics::DataFrame)
    map_data = filter(row -> row.solution == "MAP-OCS", variance_data)
    const_data = filter(row -> row.solution == "Constrained", variance_data)
    
    # Panel 1: Gain distributions
    p1 = histogram(map_data.genetic_gain, alpha=0.6, label="MAP-OCS",
                   title="Genetic Gain Distributions", xlabel="Genetic Gain",
                   ylabel="Frequency", color=:blue)
    histogram!(p1, const_data.genetic_gain, alpha=0.6, label="Constrained",
              color=:red)
    vline!([metrics[1, :mean_gain]], color=:blue, lw=2, ls=:dash, label=nothing)
    vline!([metrics[2, :mean_gain]], color=:red, lw=2, ls=:dash, label=nothing)
    
    # Panel 2: Box plots
    p2 = boxplot(["MAP-OCS"], map_data.genetic_gain, color=:blue, label="MAP-OCS",
                 title="Variability Comparison", ylabel="Genetic Gain")
    boxplot!(p2, ["Constrained"], const_data.genetic_gain, color=:red, 
             label="Constrained")
    
    # Panel 3: Gain vs Coancestry
    p3 = scatter(map_data.coancestry, map_data.genetic_gain, alpha=0.4,
                label="MAP-OCS", title="Gain-Coancestry Trade-off",
                xlabel="Coancestry", ylabel="Genetic Gain",
                color=:blue, ms=3)
    scatter!(p3, const_data.coancestry, const_data.genetic_gain, alpha=0.4,
            label="Constrained", color=:red, ms=3)
    
    # Panel 4: CV comparison
    p4 = bar(["MAP-OCS", "Constrained"], 
            [metrics[1, :cv_gain], metrics[2, :cv_gain]],
            title="Coefficient of Variation", ylabel="CV",
            color=[:blue, :red], legend=false)
    
    plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900),
         plot_title="MCMC Variance Comparison: MAP-OCS vs Constrained Exclusion")
end

"""
    create_robustness_plot(robustness_results, c_map) -> Plot

Visualize robustness scores for selected vs unselected individuals.
"""
function create_robustness_plot(robustness_results::DataFrame,
                               c_map::Vector{Float64})
    selected = filter(row -> row.selected_in_map, robustness_results)
    unselected = filter(row -> !row.selected_in_map, robustness_results)
    
    p = scatter([], [], xlabel="Breeding Value", 
               ylabel="Percentage Impact (%)",
               title="Robustness Analysis: Selected vs Replacement Candidates",
               legend=:topright, size=(900, 600))
    
    if nrow(selected) > 0
        scatter!(p, selected.breeding_value, selected.percentage_impact,
                color=:blue, ms=6, alpha=0.7, label="MAP-OCS Selected")
    end
    
    if nrow(unselected) > 0
        scatter!(p, unselected.breeding_value, unselected.percentage_impact,
                color=:orange, ms=6, alpha=0.7, 
                label="Replacement Candidates")
    end
    
    return p
end

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

"""
    run_complete_analysis() -> Dict

Execute full optimized analysis pipeline.
"""
function run_complete_analysis()
    println("="^80)
    println("OPTIMIZED MCMC ROBUSTNESS ANALYSIS - NORWAY SPRUCE")
    println("Two-Step Candidate Selection Strategy")
    println("="^80)
    
    # 1. Load data
    G, IDs, n_gav, m, n, mcmc_bv, c_map = load_and_prepare_data()
    
    # 2. Select candidates (two-step strategy)
    candidates = select_robustness_candidates(c_map, n_gav, N_REPLACEMENT_CANDIDATES)
    
    # 3. Calculate robustness scores
    robustness_results = calculate_robustness_scores(mcmc_bv, G, m, n, THETA,
                                                     candidates, c_map)
    
    # Add breeding values to results
    robustness_results.breeding_value = n_gav[robustness_results.individual]
    
    # 4. Classify high-risk individuals
    high_risk = classify_high_risk_individuals(robustness_results, c_map, 
                                               HIGH_RISK_PERCENTILE)
    
    # 5. Run constrained OCS
    println("\n" * "="^80)
    println("CONSTRAINED OCS")
    println("="^80)
    println("\nExcluding $(length(high_risk)) high-risk individuals...")
    
    c_const, g_const, coanc_const, success = OC_constrained(G, n_gav, m, n, 
                                                            THETA, high_risk)
    
    if !success
        error("Constrained OCS failed to converge")
    end
    
    n_const_selected = sum(c_const .> SELECTION_THRESHOLD)
    println("✓ Constrained OCS complete")
    println("  Selected: $n_const_selected individuals")
    println("  Genetic gain: $(round(g_const, digits=6))")
    println("  Coancestry: $(round(coanc_const, digits=6))")
    
    # 6. Variance comparison
    variance_metrics, variance_data = calculate_variance_metrics(mcmc_bv, c_map,
                                                                c_const, G)
    
    # 7. Statistical tests
    statistical_results = perform_statistical_tests(variance_data)
    
    # 8. Create visualizations
    println("\n" * "="^80)
    println("CREATING VISUALIZATIONS")
    println("="^80)
    
    p_variance = create_variance_comparison_plot(variance_data, variance_metrics)
    p_robustness = create_robustness_plot(robustness_results, c_map)
    
    # 9. Save results
    println("\n" * "="^80)
    println("SAVING RESULTS")
    println("="^80)
    
    CSV.write(joinpath(SAVE_PATH, "robustness_analysis_optimized.csv"), 
             robustness_results)
    CSV.write(joinpath(SAVE_PATH, "variance_metrics_optimized.csv"), 
             variance_metrics)
    CSV.write(joinpath(SAVE_PATH, "variance_data_optimized.csv"), 
             variance_data)
    CSV.write(joinpath(SAVE_PATH, "statistical_tests_optimized.csv"), 
             statistical_results)
    
    # Save high-risk individual list
    high_risk_df = DataFrame(individual = high_risk, 
                            excluded = true)
    CSV.write(joinpath(SAVE_PATH, "high_risk_individuals.csv"), high_risk_df)
    
    savefig(p_variance, joinpath(FIGURES_PATH, "variance_comparison_optimized.pdf"))
    savefig(p_robustness, joinpath(FIGURES_PATH, "robustness_scores_optimized.pdf"))
    
    println("✓ Results saved to: $SAVE_PATH")
    println("✓ Figures saved to: $FIGURES_PATH")
    
    # 10. Summary
    println("\n" * "="^80)
    println("ANALYSIS COMPLETE - SUMMARY")
    println("="^80)
    
    println("\nCandidates analyzed: $(length(candidates))")
    println("  MAP-OCS selected: $(sum(c_map .> SELECTION_THRESHOLD))")
    println("  Replacement pool: $N_REPLACEMENT_CANDIDATES")
    
    println("\nHigh-risk exclusions: $(length(high_risk))")
    println("  $(round(length(high_risk)/sum(c_map .> SELECTION_THRESHOLD)*100, digits=1))% " *
            "of MAP-selected")
    
    println("\nGenetic gain:")
    println("  MAP-OCS: $(round(variance_metrics[1, :mean_gain], digits=6)) ± " *
            "$(round(variance_metrics[1, :std_gain], digits=6))")
    println("  Constrained: $(round(variance_metrics[2, :mean_gain], digits=6)) ± " *
            "$(round(variance_metrics[2, :std_gain], digits=6))")
    println("  Change: $(round((variance_metrics[2, :mean_gain] - variance_metrics[1, :mean_gain]) / variance_metrics[1, :mean_gain] * 100, digits=2))%")
    
    println("\nStability (CV):")
    println("  MAP-OCS: $(round(variance_metrics[1, :cv_gain], digits=4))")
    println("  Constrained: $(round(variance_metrics[2, :cv_gain], digits=4))")
    println("  Improvement: $(round((variance_metrics[1, :cv_gain] - variance_metrics[2, :cv_gain]) / variance_metrics[1, :cv_gain] * 100, digits=2))%")
    
    # Return results dictionary
    return Dict(
        "robustness_results" => robustness_results,
        "high_risk_individuals" => high_risk,
        "c_map" => c_map,
        "c_constrained" => c_const,
        "variance_metrics" => variance_metrics,
        "variance_data" => variance_data,
        "statistical_tests" => statistical_results,
        "G" => G,
        "n_gav" => n_gav,
        "mcmc_bv" => mcmc_bv
    )
end

# ============================================================================
# EXECUTION
# ============================================================================

println("""
================================================================================
OPTIMIZED NORWAY SPRUCE MCMC ROBUSTNESS ANALYSIS
================================================================================

Ready to run! Execute:

    results = run_complete_analysis()

This will:
  1. Load data (G-matrix, breeding values, MCMC samples)
  2. Select candidates (MAP-selected + top 150 unselected by EBV)
  3. Calculate robustness scores
  4. Classify bottom 25% of MAP-selected as high-risk
  5. Run constrained OCS (excluding high-risk)
  6. Perform variance comparison across MCMC iterations
  7. Statistical hypothesis testing
  8. Create publication-quality figures
  9. Save all results

================================================================================
""")

# Uncomment to run automatically:
# results = run_complete_analysis()
