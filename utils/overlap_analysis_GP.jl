# ============================================================================
# OCS OVERLAP ANALYSIS - PLOTTING FUNCTIONS
# ============================================================================
# Standalone plotting functions that read overlap_analysis.csv files
# ============================================================================

using DataFrames
using CSV
using Plots
using StatsPlots
using Statistics
using Printf
using StatsBase
using DelimitedFiles
using AbstractGPs
using KernelFunctions
using Optim


# ============================================================================
# MAIN PLOTTING FUNCTIONS
# ============================================================================

"""
    plot_overlap_histograms(csv_file_path::String; dataset_name::String="Dataset", save_plots::Bool=true, save_path::String="")

Read overlap_analysis.csv and create comprehensive histogram plots.

# Arguments
- `csv_file_path`: Path to the overlap_analysis.csv file
- `dataset_name`: Name for the dataset (used in titles and file names)
- `save_plots`: Whether to save the plots to files
- `save_path`: Directory to save plots (if empty, saves to same directory as CSV)

# Returns
- Tuple of plot objects: (histogram_plot, trend_plot, summary_stats)
"""
function plot_overlap_histograms(csv_file_path::String; 
                                dataset_name::String="Dataset", 
                                save_plots::Bool=true, 
                                save_path::String="")
    
    # Check if file exists
    if !isfile(csv_file_path)
        error("CSV file not found: $csv_file_path")
    end
    
    println("Loading overlap analysis data from: $csv_file_path")
    
    # Read the CSV file
    data = CSV.read(csv_file_path, DataFrame)
    
    # Validate required columns
    required_cols = ["Overlap_Count", "Jaccard_Similarity", "Rank_Correlation", "Selection_Intensity_Ratio"]
    missing_cols = setdiff(required_cols, names(data))
    if !isempty(missing_cols)
        error("Missing required columns: $missing_cols")
    end
    
    println("Data loaded successfully:")
    println("  - Number of MCMC iterations: $(nrow(data))")
    println("  - Columns available: $(names(data))")
    
    # Calculate summary statistics
    stats = calculate_overlap_stats(data)
    print_summary_stats(stats, dataset_name)
    
    # Create plots
    hist_plot = create_histogram_panel(data, dataset_name, stats)
    trend_plot = create_trend_analysis(data, dataset_name, stats)
    
    # Set save path
    if isempty(save_path)
        save_path = dirname(csv_file_path)
    end
    
    # Save plots if requested
    if save_plots
        save_name = replace(lowercase(dataset_name), " " => "_")
        
        savefig(hist_plot, joinpath(save_path, "$(save_name)_overlap_histograms.png"))
        savefig(hist_plot, joinpath(save_path, "$(save_name)_overlap_histograms.pdf"))
        
        savefig(trend_plot, joinpath(save_path, "$(save_name)_trend_analysis.png"))
        savefig(trend_plot, joinpath(save_path, "$(save_name)_trend_analysis.pdf"))
        
        println("Plots saved to: $save_path")
    end
    
    return (histogram_plot=hist_plot, trend_plot=trend_plot, summary_stats=stats)
end

"""
    create_histogram_panel(data::DataFrame, dataset_name::String, stats::NamedTuple)

Create the main 4-panel histogram plot.
"""
function create_histogram_panel(data::DataFrame, dataset_name::String, stats::NamedTuple)
    
    # Plot 1: Overlap Count Distribution
    p1 = histogram(data.Overlap_Count, 
                   bins=30, 
                   title="Selection Overlap Distribution",
                   xlabel="Number of Overlapping Individuals",
                   ylabel="Frequency",
                   color=:steelblue,
                   alpha=0.7,
                   label=false,
                   linewidth=0)
    
    vline!([stats.overlap_mean], 
           color=:red, linewidth=3, label="Mean: $(round(stats.overlap_mean, digits=1))")
    vline!([stats.overlap_median], 
           color=:orange, linewidth=2, linestyle=:dash, label="Median: $(round(stats.overlap_median, digits=1))")
    
    # Add range annotation
    annotate!([(stats.overlap_max * 0.8, maximum(fit(Histogram, data.Overlap_Count, nbins=30).weights) * 0.8, 
               text("Range: [$(stats.overlap_min), $(stats.overlap_max)]", 9, :right))])
    
    # Plot 2: Jaccard Similarity Distribution
    p2 = histogram(data.Jaccard_Similarity .* 100, 
                   bins=30,
                   title="Jaccard Similarity Distribution", 
                   xlabel="Jaccard Similarity (%)",
                   ylabel="Frequency",
                   color=:coral,
                   alpha=0.7,
                   label=false,
                   linewidth=0)
    
    vline!([stats.jaccard_mean * 100], 
           color=:red, linewidth=3, label="Mean: $(round(stats.jaccard_mean*100, digits=1))%")
    vline!([stats.jaccard_median * 100], 
           color=:orange, linewidth=2, linestyle=:dash, label="Median: $(round(stats.jaccard_median*100, digits=1))%")
    
    # Plot 3: Rank Correlation Distribution
    p3 = histogram(data.Rank_Correlation, 
                   bins=30,
                   title="Rank Correlation Distribution",
                   xlabel="Spearman Rank Correlation",
                   ylabel="Frequency", 
                   color=:mediumpurple,
                   alpha=0.7,
                   label=false,
                   linewidth=0)
    
    vline!([stats.rank_corr_mean], 
           color=:red, linewidth=3, label="Mean: $(round(stats.rank_corr_mean, digits=3))")
    vline!([0], color=:gray, linestyle=:dot, linewidth=2, label="Zero correlation")
    
    # Plot 4: Selection Intensity Ratio
    p4 = histogram(data.Selection_Intensity_Ratio, 
                   bins=30,
                   title="Selection Intensity Ratio",
                   xlabel="MCMC/MAP Intensity Ratio", 
                   ylabel="Frequency",
                   color=:darkgreen,
                   alpha=0.7,
                   label=false,
                   linewidth=0)
    
    vline!([stats.intensity_mean], 
           color=:red, linewidth=3, label="Mean: $(round(stats.intensity_mean, digits=2))")
    vline!([1.0], color=:gray, linestyle=:dot, linewidth=2, label="Equal intensity")
    
    # Combine plots with overall title
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
    title = "Overlap Metrics Distribution - $dataset_name\n" *
            "Uncertainty Level: $(stats.uncertainty_level) | " *
            "Composite Score: $(round(stats.uncertainty_score, digits=3))"
    plot!(plot_title=title, plot_titlefontsize=14)
    
    return combined_plot
end

"""
    create_trend_analysis(data::DataFrame, dataset_name::String, stats::NamedTuple)

Create trend analysis and correlation plots.
"""
function create_trend_analysis(data::DataFrame, dataset_name::String, stats::NamedTuple)
    iterations = 1:nrow(data)
    
    # Plot 1: Overlap trends over iterations
    p1 = plot(iterations, data.Overlap_Count,
              title="Overlap Count Across Iterations",
              xlabel="MCMC Iteration", 
              ylabel="Overlap Count",
              color=:steelblue,
              alpha=0.6,
              linewidth=1,
              label=false)
    
    hline!([stats.overlap_mean], 
           color=:red, linestyle=:dash, linewidth=2, label="Mean: $(round(stats.overlap_mean, digits=1))")
    hline!([stats.overlap_mean - stats.overlap_std, stats.overlap_mean + stats.overlap_std], 
           color=:orange, linestyle=:dot, linewidth=1, label="±1 SD")
    
    # Plot 2: Jaccard similarity trends
    p2 = plot(iterations, data.Jaccard_Similarity .* 100,
              title="Jaccard Similarity Across Iterations", 
              xlabel="MCMC Iteration",
              ylabel="Jaccard Similarity (%)",
              color=:coral, 
              alpha=0.6,
              linewidth=1,
              label=false)
    
    hline!([stats.jaccard_mean * 100], 
           color=:red, linestyle=:dash, linewidth=2, label="Mean: $(round(stats.jaccard_mean*100, digits=1))%")
    
    # Plot 3: Scatter plot - Overlap vs Jaccard
    p3 = scatter(data.Overlap_Count, data.Jaccard_Similarity .* 100,
                 title="Overlap vs Jaccard Similarity",
                 xlabel="Overlap Count",
                 ylabel="Jaccard Similarity (%)", 
                 color=:mediumpurple,
                 alpha=0.6,
                 markersize=3,
                 label=false)
    
    # Add correlation coefficient
    corr_coef = cor(data.Overlap_Count, data.Jaccard_Similarity)
    annotate!([(stats.overlap_max * 0.2, maximum(data.Jaccard_Similarity) * 80, 
               text("r = $(round(corr_coef, digits=3))", 10, :left))])
    
    # Plot 4: Metrics correlation matrix heatmap
    if "Contribution_Correlation" in names(data) && "Kendall_Tau" in names(data)
        metrics_data = [data.Overlap_Count data.Jaccard_Similarity data.Rank_Correlation data.Contribution_Correlation]
        metrics_labels = ["Overlap", "Jaccard", "Spearman", "Pearson"]
        
        corr_matrix = cor(metrics_data)
        
        p4 = heatmap(corr_matrix,
                     title="Metrics Correlation Matrix",
                     xlabel="Metrics",
                     ylabel="Metrics",
                     xticks=(1:4, metrics_labels),
                     yticks=(1:4, metrics_labels),
                     color=:RdBu,
                     aspect_ratio=:equal)
        
        # Add correlation values as text
        for i in 1:4, j in 1:4
            annotate!([(j, i, text("$(round(corr_matrix[i,j], digits=2))", 8, :center))])
        end
    else
        # Simple metrics summary if correlation data not available
        p4 = bar(["Overlap", "Jaccard(%)", "Rank Corr", "Intensity"], 
                 [stats.overlap_mean, stats.jaccard_mean*100, stats.rank_corr_mean, stats.intensity_mean],
                 title="Summary Statistics",
                 ylabel="Value",
                 color=[:steelblue, :coral, :mediumpurple, :darkgreen],
                 alpha=0.7,
                 label=false)
    end
    
    # Combine trend plots
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
    plot!(plot_title="Trend Analysis and Correlations - $dataset_name", plot_titlefontsize=14)
    
    return combined_plot
end

"""
    calculate_overlap_stats(data::DataFrame)

Calculate comprehensive summary statistics from overlap data.
"""
function calculate_overlap_stats(data::DataFrame)
    
    # Basic statistics for each metric
    overlap_stats = (
        mean = mean(data.Overlap_Count),
        median = median(data.Overlap_Count),
        std = std(data.Overlap_Count),
        min = minimum(data.Overlap_Count),
        max = maximum(data.Overlap_Count),
        q25 = quantile(data.Overlap_Count, 0.25),
        q75 = quantile(data.Overlap_Count, 0.75)
    )
    
    jaccard_stats = (
        mean = mean(data.Jaccard_Similarity),
        median = median(data.Jaccard_Similarity),
        std = std(data.Jaccard_Similarity),
        min = minimum(data.Jaccard_Similarity),
        max = maximum(data.Jaccard_Similarity)
    )
    
    rank_corr_stats = (
        mean = mean(data.Rank_Correlation),
        median = median(data.Rank_Correlation),
        std = std(data.Rank_Correlation),
        min = minimum(data.Rank_Correlation),
        max = maximum(data.Rank_Correlation)
    )
    
    intensity_stats = (
        mean = mean(data.Selection_Intensity_Ratio),
        median = median(data.Selection_Intensity_Ratio),
        std = std(data.Selection_Intensity_Ratio)
    )
    
    # Calculate composite uncertainty score
    cv_overlap = overlap_stats.std / overlap_stats.mean
    uncertainty_score = (1 - jaccard_stats.mean) * 0.3 + 
                       (1 - abs(rank_corr_stats.mean)) * 0.3 + 
                       cv_overlap * 0.2 + 
                       abs(intensity_stats.mean - 1.0) * 0.2
    
    # Determine uncertainty level
    uncertainty_level = if uncertainty_score < 0.3
        "LOW"
    elseif uncertainty_score < 0.6
        "MODERATE" 
    else
        "HIGH"
    end
    
    return (
        # Individual metric stats
        overlap_mean = overlap_stats.mean,
        overlap_median = overlap_stats.median,
        overlap_std = overlap_stats.std,
        overlap_min = overlap_stats.min,
        overlap_max = overlap_stats.max,
        overlap_q25 = overlap_stats.q25,
        overlap_q75 = overlap_stats.q75,
        
        jaccard_mean = jaccard_stats.mean,
        jaccard_median = jaccard_stats.median,
        jaccard_std = jaccard_stats.std,
        jaccard_min = jaccard_stats.min,
        jaccard_max = jaccard_stats.max,
        
        rank_corr_mean = rank_corr_stats.mean,
        rank_corr_std = rank_corr_stats.std,
        
        intensity_mean = intensity_stats.mean,
        intensity_std = intensity_stats.std,
        
        # Composite metrics
        uncertainty_score = uncertainty_score,
        uncertainty_level = uncertainty_level,
        cv_overlap = cv_overlap
    )
end

"""
    print_summary_stats(stats::NamedTuple, dataset_name::String)

Print comprehensive summary statistics.
"""
function print_summary_stats(stats::NamedTuple, dataset_name::String)
    println("\n" * "="^80)
    println("OVERLAP ANALYSIS SUMMARY - $dataset_name")
    println("="^80)
    
    println("SELECTION OVERLAP:")
    println("  • Mean: $(round(stats.overlap_mean, digits=1)) individuals")
    println("  • Median: $(round(stats.overlap_median, digits=1)) individuals") 
    println("  • Range: [$(stats.overlap_min), $(stats.overlap_max)]")
    println("  • IQR: [$(round(stats.overlap_q25, digits=1)), $(round(stats.overlap_q75, digits=1))]")
    println("  • Coefficient of Variation: $(round(stats.cv_overlap, digits=3))")
    
    println("\nJACCARD SIMILARITY:")
    println("  • Mean: $(round(stats.jaccard_mean*100, digits=1))%")
    println("  • Range: [$(round(stats.jaccard_min*100, digits=1))%, $(round(stats.jaccard_max*100, digits=1))%]")
    println("  • Standard deviation: $(round(stats.jaccard_std*100, digits=1))%")
    
    println("\nRANK CORRELATION:")
    println("  • Mean Spearman ρ: $(round(stats.rank_corr_mean, digits=3))")
    println("  • Standard deviation: $(round(stats.rank_corr_std, digits=3))")
    
    println("\nSELECTION INTENSITY:")
    println("  • Mean ratio (MCMC/MAP): $(round(stats.intensity_mean, digits=3))")
    println("  • $(round((stats.intensity_mean-1)*100, digits=1))% $(stats.intensity_mean > 1 ? "more" : "less") concentrated than MAP")
    
    println("\nUNCERTAINTY ASSESSMENT:")
    println("  • Composite uncertainty score: $(round(stats.uncertainty_score, digits=3))")
    println("  • Uncertainty level: $(stats.uncertainty_level)")
    
    # Interpretation and recommendations
    println("\nINTERPRETATION:")
    if stats.uncertainty_level == "LOW"
        println("  🟢 Low uncertainty impact - MAP-based selection is reliable")
        println("  • Selection decisions are robust to breeding value uncertainty")
        println("  • Core selection of ~$(round(Int, stats.overlap_mean * 0.8)) individuals is highly reliable")
    elseif stats.uncertainty_level == "MODERATE"
        println("  🟡 Moderate uncertainty impact - consider hybrid approaches")
        println("  • ~$(round(Int, stats.overlap_mean)) core selections are reliable")
        println("  • Consider expanding selection pool by 20-30%")
        println("  • Focus additional data collection on marginal selections")
    else
        println("  🔴 High uncertainty impact - more data strongly recommended")
        println("  • Only ~$(round(Int, stats.overlap_mean * 0.6)) selections are highly reliable")
        println("  • Breeding value estimates need improvement before major decisions")
        println("  • Consider conservative selection strategies")
    end
    
    println("="^80)
end

"""
    compare_multiple_datasets(csv_files::Vector{String}, dataset_names::Vector{String})

Create comparative plots across multiple overlap analysis files.
"""
function compare_multiple_datasets(csv_files::Vector{String}, dataset_names::Vector{String})
    
    if length(csv_files) != length(dataset_names)
        error("Number of CSV files must match number of dataset names")
    end
    
    println("Creating comparative analysis across $(length(csv_files)) datasets...")
    
    # Load all datasets
    all_data = []
    all_stats = []
    
    for (i, csv_file) in enumerate(csv_files)
        if !isfile(csv_file)
            println("Warning: File not found: $csv_file - skipping")
            continue
        end
        
        data = CSV.read(csv_file, DataFrame)
        stats = calculate_overlap_stats(data)
        
        push!(all_data, data)
        push!(all_stats, stats)
        
        println("Loaded: $(dataset_names[i]) ($(nrow(data)) iterations)")
    end
    
    if length(all_data) < 2
        error("Need at least 2 valid datasets for comparison")
    end
    
    # Create comparison plots
    comparison_plot = create_comparison_plots(all_data, all_stats, dataset_names[1:length(all_data)])
    
    return comparison_plot, all_stats
end

"""
    create_comparison_plots(all_data, all_stats, dataset_names)

Create side-by-side comparison plots.
"""
function create_comparison_plots(all_data, all_stats, dataset_names)
    
    # Plot 1: Overlap count comparison (density plots)
    p1 = plot(title="Overlap Count Comparison", xlabel="Overlap Count", ylabel="Density")
    colors = [:steelblue, :coral, :mediumpurple, :darkgreen, :orange]
    
    for (i, data) in enumerate(all_data)
        density!(data.Overlap_Count, 
                 label=dataset_names[i],
                 linewidth=3,
                 color=colors[mod(i-1, length(colors))+1],
                 alpha=0.8)
    end
    
    # Plot 2: Jaccard similarity comparison
    p2 = plot(title="Jaccard Similarity Comparison", xlabel="Jaccard Similarity (%)", ylabel="Density")
    
    for (i, data) in enumerate(all_data)
        density!(data.Jaccard_Similarity .* 100,
                 label=dataset_names[i], 
                 linewidth=3,
                 color=colors[mod(i-1, length(colors))+1],
                 alpha=0.8)
    end
    
    # Plot 3: Summary bar chart - means
    overlap_means = [stats.overlap_mean for stats in all_stats]
    jaccard_means = [stats.jaccard_mean * 100 for stats in all_stats]
    
    p3 = groupedbar([overlap_means jaccard_means],
                    bar_position=:dodge,
                    title="Mean Values Comparison",
                    xlabel="Dataset",
                    ylabel="Value",
                    xticks=(1:length(dataset_names), dataset_names),
                    labels=["Overlap Count" "Jaccard Similarity (%)"],
                    color=[:steelblue :coral],
                    alpha=0.8)
    
    # Plot 4: Uncertainty scores comparison
    uncertainty_scores = [stats.uncertainty_score for stats in all_stats]
    uncertainty_colors = [score < 0.3 ? :green : score < 0.6 ? :orange : :red for score in uncertainty_scores]
    
    p4 = bar(dataset_names, uncertainty_scores,
             title="Uncertainty Score Comparison",
             xlabel="Dataset",
             ylabel="Composite Uncertainty Score",
             color=uncertainty_colors,
             alpha=0.8,
             label=false)
    
    hline!([0.3], color=:green, linestyle=:dash, linewidth=2, label="Low threshold")
    hline!([0.6], color=:orange, linestyle=:dash, linewidth=2, label="High threshold")
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1400, 1000))
    plot!(plot_title="Dataset Comparison - Overlap Analysis", plot_titlefontsize=16)
    
    return combined_plot
end

# ============================================================================
# SELECTION FREQUENCY VALIDATION FUNCTIONS
# ============================================================================

"""
    plot_selection_frequency_vs_ebv(contributions_csv::String, ebv_values::Vector{Float64}; 
                                   dataset_name::String="Dataset", n_selected::Int=100, 
                                   save_plots::Bool=true, save_path::String="")

Plot selection frequency across MCMC iterations vs MAP EBV values to validate results.

# Arguments
- `contributions_csv`: Path to the mcmc_contributions_matrix.csv file
- `ebv_values`: Vector of MAP EBV values for each individual
- `dataset_name`: Name for the dataset (used in titles)
- `n_selected`: Number of individuals selected per iteration (default 100)
- `save_plots`: Whether to save the plots
- `save_path`: Directory to save plots

# Returns
- Plot object and summary statistics
"""
function plot_selection_frequency_vs_ebv(contributions_csv::String, ebv_values::Vector{Float64}; 
                                        dataset_name::String="Dataset", n_selected::Int=100, 
                                        save_plots::Bool=true, save_path::String="")
    
    # Read contributions matrix
    if !isfile(contributions_csv)
        error("Contributions CSV file not found: $contributions_csv")
    end
    
    println("Loading MCMC contributions matrix from: $contributions_csv")
    contributions_matrix = readdlm(contributions_csv, ',', Float64)
    
    n_iterations, n_individuals = size(contributions_matrix)
    println("Contributions matrix: $n_iterations iterations × $n_individuals individuals")
    
    # Validate EBV vector length
    if length(ebv_values) != n_individuals
        error("EBV vector length ($(length(ebv_values))) doesn't match number of individuals ($n_individuals)")
    end
    
    # Calculate selection frequencies
    selection_frequencies = calculate_selection_frequencies(contributions_matrix, n_selected)
    
    # Create validation plots
    main_plot = create_ebv_frequency_plot(ebv_values, selection_frequencies, dataset_name, n_selected)
    detail_plot = create_detailed_validation_plots(ebv_values, selection_frequencies, contributions_matrix, dataset_name, n_selected)
    
    # Calculate validation statistics
    validation_stats = calculate_validation_statistics(ebv_values, selection_frequencies, n_selected)
    
    # Calculate Spearman correlation here where we have access to the data
    spearman_correlation = corspearman(ebv_values, selection_frequencies)
    
    print_validation_summary(validation_stats, dataset_name, spearman_correlation)
    
    # Save plots
    if save_plots
        if isempty(save_path)
            save_path = dirname(contributions_csv)
        end
        
        save_name = replace(lowercase(dataset_name), " " => "_")
        savefig(main_plot, joinpath(save_path, "$(save_name)_selection_frequency_vs_ebv.png"))
        savefig(main_plot, joinpath(save_path, "$(save_name)_selection_frequency_vs_ebv.pdf"))
        savefig(detail_plot, joinpath(save_path, "$(save_name)_detailed_validation.png"))
        savefig(detail_plot, joinpath(save_path, "$(save_name)_detailed_validation.pdf"))
        
        println("Validation plots saved to: $save_path")
    end
    
    return (main_plot=main_plot, detail_plot=detail_plot, validation_stats=validation_stats)
end

"""
    calculate_selection_frequencies(contributions_matrix::Matrix{Float64}, n_selected::Int)

Calculate how often each individual is selected across MCMC iterations.
"""
function calculate_selection_frequencies(contributions_matrix::Matrix{Float64}, n_selected::Int)
    n_iterations, n_individuals = size(contributions_matrix)
    selection_frequencies = zeros(n_individuals)
    
    for i in 1:n_iterations
        # Get top n_selected individuals for this iteration
        sorted_indices = sortperm(contributions_matrix[i, :], rev=true)
        top_selected = sorted_indices[1:n_selected]
        
        # Mark these individuals as selected
        for idx in top_selected
            selection_frequencies[idx] += 1
        end
    end
    
    # Convert to proportions
    selection_frequencies ./= n_iterations
    
    return selection_frequencies
end

"""
    create_ebv_frequency_plot(ebv_values, selection_frequencies, dataset_name, n_selected)

Create the main EBV vs selection frequency plot.
"""
function create_ebv_frequency_plot(ebv_values::Vector{Float64}, selection_frequencies::Vector{Float64}, 
                                  dataset_name::String, n_selected::Int)
    
    # Sort individuals by EBV for better visualization
    sorted_indices = sortperm(ebv_values)
    sorted_ebvs = ebv_values[sorted_indices]
    sorted_frequencies = selection_frequencies[sorted_indices]
    
    # Create the main barplot
    p = bar(1:length(sorted_ebvs), sorted_frequencies,
            title="Selection Frequency vs MAP EBV Rank - $dataset_name",
            xlabel="Individual (Ranked by MAP EBV, Low → High)",
            ylabel="Selection Frequency (Proportion)",
            color=:steelblue,
            alpha=0.7,
            label="Selection Frequency",
            linewidth=0)
    
    # Add reference lines
    expected_frequency = n_selected / length(ebv_values)
    hline!([expected_frequency], 
           color=:red, linestyle=:dash, linewidth=2, 
           label="Expected if Random ($(round(expected_frequency, digits=3)))")
    
    hline!([0.5], 
           color=:orange, linestyle=:dot, linewidth=2, 
           label="50% Selection Rate")
    
    # Highlight the top individuals that should be frequently selected
    top_n_threshold = length(ebv_values) - n_selected
    vline!([top_n_threshold], 
           color=:green, linestyle=:dashdot, linewidth=2,
           label="Top $(n_selected) EBV Threshold")
    
    # Add text annotations for key regions
    annotate!([(length(ebv_values) * 0.1, 0.9, 
               text("Low EBV\n(Rarely Selected)", 10, :center))])
    annotate!([(length(ebv_values) * 0.9, 0.9, 
               text("High EBV\n(Frequently Selected)", 10, :center))])
    
    # Customize plot
    plot!(size=(1200, 600),
          xlims=(0, length(ebv_values)),
          ylims=(0, 1),
          legend=:topleft)
    
    return p
end

"""
    create_detailed_validation_plots(ebv_values, selection_frequencies, contributions_matrix, dataset_name, n_selected)

Create detailed validation plots showing multiple perspectives.
"""
function create_detailed_validation_plots(ebv_values::Vector{Float64}, selection_frequencies::Vector{Float64}, 
                                        contributions_matrix::Matrix{Float64}, dataset_name::String, n_selected::Int)
    
    # Plot 1: Scatter plot EBV vs Selection Frequency
    p1 = scatter(ebv_values, selection_frequencies,
                 #title="EBV vs Selection Frequency",
                 xlabel="MAP EBV Index Value", 
                 ylabel="Selection Frequency",
                 color=:steelblue,
                 alpha=0.6,
                 markersize=3,
                 legend=:bottomright,
                 label=false,
                 xtickfontsize=11,    # ← Add this
                 ytickfontsize=11)
    
    # Add Gaussian Process fit (Matérn 3/2 with optimized hyperparameters)
    # Uses: AbstractGPs, KernelFunctions, Optim (loaded at top of file)
    
    # Normalize data
    x_mean, x_std = mean(ebv_values), std(ebv_values)
    y_mean, y_std = mean(selection_frequencies), std(selection_frequencies)
    x_norm = (ebv_values .- x_mean) ./ x_std
    y_norm = (selection_frequencies .- y_mean) ./ y_std
    
    # Define negative log marginal likelihood for optimization
    function neg_log_marginal_likelihood(params)
        lengthscale = exp(params[1])
        signal_var = exp(params[2])
        noise_var = exp(params[3])
        
        # Bounds checking
        if lengthscale < 0.01 || lengthscale > 100.0 || 
           signal_var < 0.001 || signal_var > 100.0 || 
           noise_var < 0.0001 || noise_var > 1.0
            return Inf
        end
        
        try
            kernel = signal_var * Matern32Kernel() ∘ ScaleTransform(lengthscale)
            f = GP(kernel)
            fx = f(x_norm, noise_var)
            return -logpdf(fx, y_norm)
        catch
            return Inf
        end
    end
    
    # Optimize hyperparameters
    initial_params = [log(1.0), log(1.0), log(0.01)]
    result = optimize(neg_log_marginal_likelihood, initial_params, NelderMead(),
                     Optim.Options(show_trace=false, iterations=1000))
    
    # Extract optimal parameters
    opt_params = Optim.minimizer(result)
    lengthscale_opt = exp(opt_params[1])
    signal_var_opt = exp(opt_params[2])
    noise_var_opt = exp(opt_params[3])
    
    # Fit GP with optimal parameters
    kernel_opt = signal_var_opt * Matern32Kernel() ∘ ScaleTransform(lengthscale_opt)
    f_opt = GP(kernel_opt)
    fx_opt = f_opt(x_norm, noise_var_opt)
    p_fx_opt = posterior(fx_opt, y_norm)
    
    # Create smooth grid for plotting
    x_grid = range(minimum(ebv_values) - 2, maximum(ebv_values) + 2, length=300)
    x_grid_norm = (x_grid .- x_mean) ./ x_std
    
    # Get predictions on grid
    y_pred_norm_grid = mean(p_fx_opt(x_grid_norm, noise_var_opt))
    y_std_norm_grid = sqrt.(var(p_fx_opt(x_grid_norm, noise_var_opt)))
    
    y_pred_grid = y_pred_norm_grid .* y_std .+ y_mean
    y_std_grid = y_std_norm_grid .* y_std
    
    # Plot GP mean with uncertainty band
    plot!(p1, x_grid, y_pred_grid, ribbon=2 .* y_std_grid,
         fillalpha=0.15, fillcolor=:purple,
         linewidth=3, color=:purple, label="GP Matérn 3/2 ± 2σ")
    
    # Calculate R² for annotation
    y_pred_norm = mean(p_fx_opt(x_norm, noise_var_opt))
    y_pred = y_pred_norm .* y_std .+ y_mean
    r2_gp = 1 - sum((selection_frequencies .- y_pred).^2) / 
                sum((selection_frequencies .- mean(selection_frequencies)).^2)
    
    # Calculate and display Spearman correlation with optimized parameters
    correlation_spearman = corspearman(ebv_values, selection_frequencies)
    annotation_text = @sprintf("ρ = %.3f\nR² = %.3f\nσₙ² = %.3f", 
                              correlation_spearman, r2_gp, noise_var_opt)
    annotate!(p1, [(-4, 0.6, 
               text(annotation_text, 10, :left))])
    
    # Plot 2: Selection frequency barplot (corrected)
    # Sort individuals by EBV for proper ordering
    sorted_indices = sortperm(ebv_values)
    sorted_frequencies = selection_frequencies[sorted_indices]
    
    p2 = bar(1:length(ebv_values), sorted_frequencies,
             #title="Selection Frequency vs MAP EBV Rank",
             xlabel="Individual (Ranked by MAP EBV, Low → High)",
             ylabel="Selection Frequency",
             color=:steelblue,
             alpha=0.7,
             label=false,
             linewidth=0,
             legend=:topright,
             xtickfontsize=11,    # ← Add this
             ytickfontsize=11)
    
    # Add reference lines
    expected_freq = n_selected / length(ebv_values)
    hline!([expected_freq], color=:red, linestyle=:dash, linewidth=2, 
           label="Expected if Random")
    
    # Mark the theoretical top n_selected EBV threshold
    vline!([length(ebv_values) - n_selected], color=:green, linestyle=:dashdot, linewidth=2,
           label="Top $(n_selected) EBV Threshold")
    
    # Calculate and display correlation
    correlation = cor(ebv_values, selection_frequencies)
    
    # Remove the annotation from the plot
    # annotate!([(length(ebv_values) * 0.8, maximum(sorted_frequencies) * 0.9, 
    #            text("r = $(round(correlation, digits=3))", 10, :left))])
    
    # Remove x-axis tick labels for cleaner appearance
    xticks!(1:div(length(ebv_values),5):length(ebv_values), repeat([""], length(1:div(length(ebv_values),5):length(ebv_values))))
    
    # Set appropriate y-axis range
    ylims!(0, 0.7)
    
    # Plot 3: Top vs Bottom EBV quartiles
    n_quartile = div(length(ebv_values), 4)
    sorted_indices = sortperm(ebv_values)
    
    bottom_quartile_freq = mean(selection_frequencies[sorted_indices[1:n_quartile]])
    top_quartile_freq = mean(selection_frequencies[sorted_indices[end-n_quartile+1:end]])
    
    quartile_data = [bottom_quartile_freq, top_quartile_freq]
    quartile_labels = ["Bottom 25%\n(Lowest EBV)", "Top 25%\n(Highest EBV)"]
    
    p3 = bar(quartile_labels, quartile_data,
             #title="Selection Rate: Top vs Bottom EBV Quartiles",
             ylabel="Mean Selection Frequency",
             color=[:lightcoral, :lightgreen],
             alpha=0.8,
             label=false,
             legend=:topright,
             xtickfontsize=11,    # ← Add this
             ytickfontsize=11)
    
    hline!([expected_freq], color=:red, linestyle=:dash, label="Expected (Random)")
    
    # Add values on top of bars
    for (i, val) in enumerate(quartile_data)
        annotate!([(i, val + 0.05, text("$(round(val, digits=3))", 10, :center))])
    end
    
    # Plot 4: Heatmap of selection patterns (sample)
    # Show selection pattern for a subset of iterations to avoid overcrowding
    n_show_iterations = min(50, size(contributions_matrix, 1))
    iteration_indices = round.(Int, range(1, size(contributions_matrix, 1), length=n_show_iterations))
    
    # Create binary selection matrix for visualization
    selection_matrix = zeros(Int, n_show_iterations, length(ebv_values))
    for (plot_row, actual_iter) in enumerate(iteration_indices)
        sorted_contrib_indices = sortperm(contributions_matrix[actual_iter, :], rev=true)
        top_selected = sorted_contrib_indices[1:n_selected]
        selection_matrix[plot_row, top_selected] .= 1
    end
    
    # Sort columns by EBV for better visualization
    sorted_indices = sortperm(ebv_values)
    selection_matrix_sorted = selection_matrix[:, sorted_indices]
    
    p4 = heatmap(selection_matrix_sorted,
               #  title="Selection Patterns (Sample of Iterations)",
                 xlabel="Individual (Ranked by EBV)",
                 ylabel="MCMC Iteration (Sample)",
                 color=[:white, :blue],
                 aspect_ratio=:auto,
                 colorbar=false,
                 xtickfontsize=12,    # ← Add this
                 ytickfontsize=12)
    
    # Add a text annotation to explain the colors since we removed the colorbar
    annotate!([(size(selection_matrix_sorted, 2) * 0.8, size(selection_matrix_sorted, 1) * 0.9, 
               text("White = Not Selected\nBlue = Selected", 8, :left))])
    
    
    
    # Add panel labels (a, b, c, d) consistently in top-left corner
    # All positioned at 2% from left edge, 96% from bottom for perfect alignment
    
    # Panel (a) - EBV vs Selection Frequency
    x1 = -5 #minimum(ebv_values) #+ 0.02 * (maximum(ebv_values) - minimum(ebv_values))
    y1 = 0.77 #minimum(selection_frequencies) + 0.99 * (maximum(selection_frequencies) - minimum(selection_frequencies))
    annotate!(p1, [(x1, y1, text("a", :left, :bold, 18))])
    
    # Panel (b) - Selection Frequency Bar Plot
    x2 = 0.0005 * length(ebv_values)
    y2 = 0.66  # y-axis fixed at 0 to 1
    annotate!(p2, [(x2, y2, text("b", :left, :bold, 18))])
    
    # Panel (c) - Quartile Comparison
    x3 = 0.04  # x-axis is categories 0-1
    y3 = 0.95 * maximum(quartile_data)
    annotate!(p3, [(x3, y3, text("c", :left, :bold, 18))])
    
    # Panel (d) - Selection Pattern Heatmap
    x4 = 0.04 * size(selection_matrix_sorted, 2)
    y4 = 0.96 * size(selection_matrix_sorted, 1)
    annotate!(p4, [(x4, y4, text("d", :left, :bold, 18))])
    # Combine all plots
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1200))
 #   plot!(plot_title="Detailed Validation Analysis - $dataset_name", plot_titlefontsize=14)
    
    return combined_plot
end
"""
    calculate_validation_statistics(ebv_values, selection_frequencies, n_selected)

Calculate comprehensive validation statistics.
"""
function calculate_validation_statistics(ebv_values::Vector{Float64}, selection_frequencies::Vector{Float64}, n_selected::Int)
    
    # Basic correlation
    correlation_ebv_freq = cor(ebv_values, selection_frequencies)
    
    # Top vs bottom quartile analysis
    n_quartile = div(length(ebv_values), 4)
    sorted_indices = sortperm(ebv_values)
    
    bottom_quartile_indices = sorted_indices[1:n_quartile]
    top_quartile_indices = sorted_indices[end-n_quartile+1:end]
    
    bottom_quartile_freq = mean(selection_frequencies[bottom_quartile_indices])
    top_quartile_freq = mean(selection_frequencies[top_quartile_indices])
    quartile_ratio = top_quartile_freq / (bottom_quartile_freq + 1e-10)  # Avoid division by zero
    
    # Expected vs actual for top performers
    expected_frequency = n_selected / length(ebv_values)
    top_n_indices = sorted_indices[end-n_selected+1:end]  # Top n_selected by EBV
    top_n_mean_frequency = mean(selection_frequencies[top_n_indices])
    
    # Count how many of top EBV individuals are frequently selected (>50%)
    top_frequent_selections = sum(selection_frequencies[top_n_indices] .> 0.5)
    
    # Validation score (0-1, where 1 is perfect)
    validation_score = (correlation_ebv_freq + 1) / 2 * 0.4 +  # Correlation component
                      min(quartile_ratio / 10, 1) * 0.3 +        # Quartile ratio component  
                      min(top_n_mean_frequency / 0.8, 1) * 0.3   # Top performers component
    
    return (
        correlation = correlation_ebv_freq,
        bottom_quartile_freq = bottom_quartile_freq,
        top_quartile_freq = top_quartile_freq,
        quartile_ratio = quartile_ratio,
        expected_frequency = expected_frequency,
        top_n_mean_frequency = top_n_mean_frequency,
        top_frequent_selections = top_frequent_selections,
        validation_score = validation_score,
        n_individuals = length(ebv_values),
        n_selected = n_selected
    )
end

"""
    print_validation_summary(stats, dataset_name, spearman_correlation)

Print comprehensive validation summary.
"""
function print_validation_summary(stats, dataset_name::String, spearman_correlation::Float64)
    println("\n" * "="^80)
    println("SELECTION VALIDATION SUMMARY - $dataset_name")
    println("="^80)
    
    println("BASIC VALIDATION METRICS:")
    println("  • EBV-Frequency Correlation (Pearson): $(round(stats.correlation, digits=3))")
    println("  • EBV-Frequency Correlation (Spearman): $(round(spearman_correlation, digits=3))")
    
    println("  • Expected selection frequency: $(round(stats.expected_frequency, digits=3))")
    println("  • Top $(stats.n_selected) EBV mean frequency: $(round(stats.top_n_mean_frequency, digits=3))")
    
    println("\nQUARTILE ANALYSIS:")
    println("  • Bottom 25% EBV mean frequency: $(round(stats.bottom_quartile_freq, digits=3))")
    println("  • Top 25% EBV mean frequency: $(round(stats.top_quartile_freq, digits=3))")
    println("  • Top/Bottom ratio: $(round(stats.quartile_ratio, digits=1))x")
    
    println("\nTOP PERFORMER ANALYSIS:")
    println("  • Top $(stats.n_selected) individuals frequently selected (>50%): $(stats.top_frequent_selections)/$(stats.n_selected)")
    println("  • Percentage of top EBV frequently selected: $(round(stats.top_frequent_selections/stats.n_selected*100, digits=1))%")
    
    println("\nVALIDATION ASSESSMENT:")
    println("  • Composite validation score: $(round(stats.validation_score, digits=3)) (0-1 scale)")
    
    # Interpretation
    if stats.correlation > 0.7 && stats.quartile_ratio > 5
        println("  🟢 EXCELLENT: Strong positive correlation, clear EBV-selection relationship")
        println("    → Results are biologically sensible and statistically sound")
    elseif stats.correlation > 0.4 && stats.quartile_ratio > 2
        println("  🟡 GOOD: Moderate correlation, some EBV-selection relationship")
        println("    → Results show expected pattern but with significant uncertainty")
    elseif stats.correlation > 0.2
        println("  🟠 CONCERNING: Weak correlation, limited EBV-selection relationship")
        println("    → High uncertainty is affecting selection patterns")
    else
        println("  🔴 PROBLEMATIC: No/negative correlation, EBV-selection relationship unclear")
        println("    → Results may not be reliable, investigate data quality")
    end
    
    # Specific recommendations
    println("\nRECOMMENDATION:")
    if stats.validation_score > 0.7
        println("  • Selection patterns validate well - proceed with confidence")
        println("  • Focus breeding decisions on consistently selected individuals")
    elseif stats.validation_score > 0.4
        println("  • Moderate validation - use results with caution")
        println("  • Consider expanding selection pool to account for uncertainty")
    else
        println("  • Poor validation - more data strongly recommended")
        println("  • Review breeding value estimation methodology")
    end
    
    println("="^80)
end

"""
    load_ebv_from_contributions_directory(contributions_csv_path::String)

Helper function to automatically find and load EBV values from the same directory structure.
Looks for reference EBV files in the same directory as the contributions matrix.
"""
function load_ebv_from_contributions_directory(contributions_csv_path::String)
    base_dir = dirname(contributions_csv_path)
    
    # Look for common EBV file patterns
    possible_ebv_files = [
        joinpath(base_dir, "reference_ebvs.csv"),
        joinpath(base_dir, "map_ebvs.csv"),
        joinpath(base_dir, "ebv_values.csv")
    ]
    
    for ebv_file in possible_ebv_files
        if isfile(ebv_file)
            println("Found EBV file: $ebv_file")
            ebv_data = readdlm(ebv_file, ',', Float64)
            return vec(ebv_data)  # Convert to vector if needed
        end
    end
    
    println("No EBV file found automatically. Please provide EBV values manually.")
    return nothing
end

"""
    quick_plot(csv_file_path::String; dataset_name::String="Dataset")

Quick function to create all plots from a CSV file.
"""
function quick_plot(csv_file_path::String; dataset_name::String="Dataset")
    return plot_overlap_histograms(csv_file_path; dataset_name=dataset_name, save_plots=true)
end

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# Example 1: Plot single dataset
 csv_path = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Save\\overlap_analysis.csv"
 results = plot_overlap_histograms(csv_path; dataset_name="Norway Spruce (G-matrix)")

# Example 2: Quick plot
 quick_plot(csv_path; dataset_name="Norway Spruce (G-matrix)")

# Example 3: Compare multiple datasets
 csv_files = [
     "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Save\\overlap_analysis.csv",
    "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Save\\overlap_analysis_A_1218.csv",
     "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Save\\overlap_analysis_H_5525.csv", 
     "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Save\\overlap_analysis_A_5525.csv"
 ]
 dataset_names = ["Spruce (G-matrix, 1218 ind)", "Spruce (A-matrix, 1218 ind)", "Spruce (H-matrix, 5525 ind)", "Spruce (A-matrix, 5525 ind)"]
 comparison_plot, stats = compare_multiple_datasets(csv_files, dataset_names)

 # Make sure you're calling the right function:
contributions_path = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Save\\mcmc_contributions_matrix.csv"
############ calculate EBV Index
filename2 = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\results_JWAS_1218_G_adj_Lev17\\EBV_Hjd17.txt"
Hjd17 = DataFrame(CSV.File(filename2)) # readdlm(filename2,',', header=true) 
filename3 = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\results_JWAS_1218_G_adj_Lev17\\EBV_Htv17.txt"
Htv17 = DataFrame(CSV.File(filename3)) 
filename4 = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\results_JWAS_1218_G_adj_Lev17\\EBV_Sprant17.txt"
Sprant17 = DataFrame(CSV.File(filename4))

# Initialize an empty vector 'gav'
gav = Hjd17[:,2] 

# Compute row-wise averages for 'EBV' column
for i in 1:size(Hjd17, 1)
  gav[i] = (Hjd17[i,2] + Htv17[i,2] - Sprant17[i,2])/3. # g[i] = 
end

# Normalize 'gav' to have mean 0 and standard deviation 1
n_gav = (gav .- mean(gav)) ./ std(gav)


validation_results = plot_selection_frequency_vs_ebv(contributions_path, n_gav; 
                                                     dataset_name="Norway Spruce (G-matrix)")

# Then check the detail_plot:
display(validation_results.detail_plot)