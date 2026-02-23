using DataFrames, CSV, Plots, StatsPlots, Statistics, KernelDensity
using Colors, PlotThemes, Measures, DelimitedFiles

# Set up paths for your JWAS output
base_path = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Save\\"

# Define MCMC file names for each trait
mcmc_files = Dict(
    "Hjd17" => joinpath(base_path, "MCMC_samples_EBV_Hjd17.txt"),
    "Hjd7" => joinpath(base_path, "MCMC_samples_EBV_Hjd7.txt"),
    "Htv17" => joinpath(base_path, "MCMC_samples_EBV_Htv17.txt"),
    "Sprant17" => joinpath(base_path, "MCMC_samples_EBV_Sprant17.txt"),
    "Lev17" => joinpath(base_path, "MCMC_samples_EBV_Lev17.txt")
)

# Load family and individual ID information from your phenotype file
phenotype_file = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Save\\JWAS_phenotypes_1218.txt"

println("Loading individual and family information...")
phenotype_data = CSV.read(phenotype_file, DataFrame)

# Extract ID and Family information
family_data = select(phenotype_data, :ID, :Famly)
rename!(family_data, :Famly => :Family_ID)  # Standardize column name

println("Loaded $(nrow(family_data)) individuals from $(length(unique(family_data.Family_ID))) families")
println("Individual ID range: $(minimum(family_data.ID)) to $(maximum(family_data.ID))")
println("Family ID range: $(minimum(family_data.Family_ID)) to $(maximum(family_data.Family_ID))")

# Helper function to get family ID for an individual
function get_family_id(individual_id, family_data)
    idx = findfirst(x -> x == individual_id, family_data.ID)
    return idx !== nothing ? family_data.Family_ID[idx] : missing
end

# Function to load MCMC samples from JWAS output (corrected for simple header format)
function load_jwas_mcmc_samples_with_ids(mcmc_files, family_data)
    println("Loading MCMC samples from JWAS output...")
    
    mcmc_data = Dict()
    individual_ids = nothing
    
    # Load each trait's MCMC samples
    for (trait, filepath) in mcmc_files
        println("Loading $trait from: $filepath")
        
        # Check if file exists
        if !isfile(filepath)
            println("Warning: File not found: $filepath")
            continue
        end
        
        try
            # Read file line by line
            lines = readlines(filepath)
            
            if length(lines) < 2
                println("  File appears to be empty or too short")
                continue
            end
            
            # Parse first line to get individual IDs
            header_line = lines[1]
            id_strings = split(header_line, ',')
            
            if individual_ids === nothing
                # All parts are individual IDs
                individual_ids = [parse(Int, strip(id_str)) for id_str in id_strings]
                println("  Individual IDs extracted: $(length(individual_ids)) individuals")
                println("  ID range: $(minimum(individual_ids)) to $(maximum(individual_ids))")
                println("  First 5 IDs: $(individual_ids[1:min(5, length(individual_ids))])")
            end
            
            # Parse data lines (all values are EBV samples)
            n_samples = length(lines) - 1  # Exclude header
            n_individuals = length(individual_ids)
            mcmc_matrix = zeros(Float64, n_samples, n_individuals)
            
            for (i, line) in enumerate(lines[2:end])  # Skip header line
                values = split(line, ',')
                
                for (j, val_str) in enumerate(values)
                    if j <= n_individuals
                        mcmc_matrix[i, j] = parse(Float64, strip(val_str))
                    end
                end
            end
            
            mcmc_data[trait] = mcmc_matrix
            println("  Successfully loaded: $(size(mcmc_matrix)) [samples × individuals]")
            
        catch e
            println("  Error loading $trait: $e")
            continue
        end
    end
    
    # Match individual IDs with family information
    if individual_ids !== nothing && length(mcmc_data) > 0
        println("\nMatching MCMC individual IDs with family data...")
        println("MCMC IDs (first 5): $(individual_ids[1:min(5, length(individual_ids))])")
        println("Family data IDs (first 5): $(family_data.ID[1:min(5, length(family_data.ID))])")
        
        # Create mapping between MCMC individual IDs and family data
        family_mapping = DataFrame(
            Individual_ID = individual_ids,
            Family_ID = [get_family_id(id, family_data) for id in individual_ids]
        )
        
        # Remove individuals without family information
        valid_mask = .!ismissing.(family_mapping.Family_ID)
        n_valid = sum(valid_mask)
        n_missing = sum(.!valid_mask)
        
        if n_missing > 0
            println("Warning: $n_missing individuals have no family information")
            if n_missing <= 20  # Only show if not too many
                missing_ids = individual_ids[.!valid_mask]
                println("Missing IDs: $(missing_ids)")
            else
                missing_ids = individual_ids[.!valid_mask]
                println("Missing IDs (first 10): $(missing_ids[1:10])")
            end
        end
        
        family_mapping = family_mapping[valid_mask, :]
        
        # Filter MCMC data to include only individuals with family information
        filtered_mcmc_data = Dict()
        for (trait, mcmc_matrix) in mcmc_data
            filtered_mcmc_data[trait] = mcmc_matrix[:, valid_mask]
        end
        
        println("Final dataset: $n_valid individuals with complete information")
        
        return filtered_mcmc_data, family_mapping
    else
        error("No MCMC data could be loaded or no individual IDs found")
    end
end

# Load your MCMC samples with proper ID matching
mcmc_samples, family_mapping = load_jwas_mcmc_samples_with_ids(mcmc_files, family_data)

# Check which traits were successfully loaded
available_traits = collect(keys(mcmc_samples))
println("Successfully loaded traits: $available_traits")

if length(available_traits) == 0
    error("No MCMC samples could be loaded. Please check file paths and formats.")
end

# Get dimensions from first available trait
first_trait = available_traits[1]
n_mcmc_samples, n_individuals = size(mcmc_samples[first_trait])
println("MCMC dimensions: $n_mcmc_samples samples × $n_individuals individuals")

# Display family structure statistics
family_stats_initial = combine(groupby(family_mapping, :Family_ID), nrow => :N_Individuals)
sort!(family_stats_initial, :N_Individuals, rev=true)

println("\nFamily structure summary:")
println("Total families: $(nrow(family_stats_initial))")
println("Largest families (top 10):")
for i in 1:min(10, nrow(family_stats_initial))
    row = family_stats_initial[i, :]
    println("  Family $(row.Family_ID): $(row.N_Individuals) individuals")
end

# Calculate your Norway spruce selection index MCMC samples
# Your index: Hjd17 + Htv17 - Sprant17 (excluding Lev17 due to no variation in subset)
function calculate_selection_index_mcmc(mcmc_samples)
    required_traits = ["Hjd17", "Htv17", "Sprant17"]
    missing_traits = [t for t in required_traits if !(t in keys(mcmc_samples))]
    
    if length(missing_traits) > 0
        println("Warning: Missing traits for selection index: $missing_traits")
        println("Available traits: $(keys(mcmc_samples))")
        
        # Use available traits only
        if "Hjd17" in keys(mcmc_samples) && "Htv17" in keys(mcmc_samples)
            println("Calculating index with Hjd17 + Htv17 only")
            return mcmc_samples["Hjd17"] .+ mcmc_samples["Htv17"]
        else
            error("Cannot calculate selection index - insufficient traits available")
        end
    else
        # Full index calculation
        return mcmc_samples["Hjd17"] .+ mcmc_samples["Htv17"] .- mcmc_samples["Sprant17"]
    end
end

# Calculate selection index MCMC samples
println("Calculating selection index MCMC samples...")
index_mcmc = calculate_selection_index_mcmc(mcmc_samples)
println("Selection index calculated: $(size(index_mcmc))")

# Create summary statistics from MCMC samples
function create_ebv_summary_from_mcmc(mcmc_samples, index_mcmc, family_mapping)
    n_individuals = size(index_mcmc, 2)
    
    summary_df = DataFrame(
        Individual_ID = family_mapping.Individual_ID,
        Family_ID = family_mapping.Family_ID,
        Index_Mean = [mean(index_mcmc[:, i]) for i in 1:n_individuals],
        Index_SD = [std(index_mcmc[:, i]) for i in 1:n_individuals]
    )
    
    # Add individual trait summaries for available traits
    for trait in keys(mcmc_samples)
        summary_df[!, Symbol("$(trait)_Mean")] = [mean(mcmc_samples[trait][:, i]) for i in 1:n_individuals]
        summary_df[!, Symbol("$(trait)_SD")] = [std(mcmc_samples[trait][:, i]) for i in 1:n_individuals]
    end
    
    return summary_df
end

ebv_summary = create_ebv_summary_from_mcmc(mcmc_samples, index_mcmc, family_mapping)
println("Created EBV summary statistics")

# Calculate family statistics and select representative families
family_stats = combine(groupby(ebv_summary, :Family_ID),
    :Index_Mean => mean => :Family_Mean_Index,
    :Index_Mean => length => :N_Individuals,
    :Index_SD => mean => :Avg_Uncertainty
)

# Filter families with adequate sample sizes for good density plots
min_family_size = 10  # Adjust based on your family sizes
large_families = filter(row -> row.N_Individuals >= min_family_size, family_stats)

println("Families with ≥$min_family_size individuals: $(nrow(large_families))")

if nrow(large_families) < 3
    println("Warning: Few large families available. Reducing minimum family size...")
    min_family_size = 5
    large_families = filter(row -> row.N_Individuals >= min_family_size, family_stats)
    println("Families with ≥$min_family_size individuals: $(nrow(large_families))")
end

# Select three representative families
sort!(large_families, :Family_Mean_Index)

selected_families = if nrow(large_families) >= 3
    [
        large_families[end, :Family_ID],      # High performance
        large_families[div(end,2), :Family_ID], # Medium performance
        large_families[1, :Family_ID]         # Low performance
    ]
else
    # Take what we have
    large_families[1:min(3, nrow(large_families)), :Family_ID]
end

family_labels = length(selected_families) == 3 ? 
    ["High Performance", "Medium Performance", "Low Performance"] :
    ["Family $(i)" for i in 1:length(selected_families)]

println("Selected families: $selected_families")
println("Family sample sizes: ", [sum(ebv_summary.Family_ID .== f) for f in selected_families])

# Function to calculate ranking uncertainty within family
function calculate_ranking_uncertainty(family_mcmc_matrix)
    n_samples, n_individuals = size(family_mcmc_matrix)
    
    if n_individuals < 2
        return 0.0
    end
    
    # Calculate ranking for each MCMC sample
    rankings = zeros(Int, n_samples, n_individuals)
    
    for s in 1:n_samples
        sample_values = family_mcmc_matrix[s, :]
        rankings[s, :] = sortperm(sample_values, rev=true)
    end
    
    # Calculate ranking variance (average standard deviation of ranks)
    rank_uncertainties = [std(rankings[:, i]) for i in 1:n_individuals]
    
    return mean(rank_uncertainties)
end

# Function to create MCMC uncertainty plots for Norway spruce families
function plot_norway_spruce_mcmc_uncertainty(mcmc_samples, index_mcmc, ebv_summary, 
                                            families, labels; trait="Index", n_top_individuals=5)
    
    colors = [:steelblue, :darkgreen, :coral]
    alpha_individual = 0.3
    alpha_top = 0.8
    
    plots_array = []
    
    for (i, (fam, label)) in enumerate(zip(families, labels))
        # Get family data
        family_mask = ebv_summary.Family_ID .== fam
        family_indices = findall(family_mask)
        family_summary = filter(row -> row.Family_ID == fam, ebv_summary)
        
        # Sort individuals by mean EBV (descending for selection index)
        sort!(family_summary, Symbol("$(trait)_Mean"), rev=true)
        
        # Get MCMC samples for this family
        if trait == "Index"
            family_mcmc = index_mcmc[:, family_indices]
        elseif trait in keys(mcmc_samples)
            family_mcmc = mcmc_samples[trait][:, family_indices]
        else
            println("Warning: Trait $trait not available")
            continue
        end
        
        # Create plot
        p = plot(title="$label (n=$(length(family_indices)))",
                xlabel=i == 2 ? "$trait EBV" : "",
                ylabel=i == 1 ? "Density" : "",
                titlefontsize=12,
                legend=false)
        
        # Plot all individuals with low alpha
        for j in 1:size(family_mcmc, 2)
            individual_samples = family_mcmc[:, j]
            
            # Create density for this individual
            try
                density_obj = kde(individual_samples)
                
                # Plot with low alpha for all individuals
                plot!(p, density_obj.x, density_obj.density,
                     color=colors[i],
                     alpha=alpha_individual,
                     linewidth=1)
            catch e
                # Skip individuals with insufficient variation
                continue
            end
        end
        
        # Highlight top performers with higher alpha and thicker lines
        top_n = min(n_top_individuals, nrow(family_summary))
        top_indices = family_summary[1:top_n, :Individual_ID]
        
        for top_id in top_indices
            # Find the position of this individual in the family
            original_idx = findfirst(x -> x == top_id, ebv_summary.Individual_ID)
            if original_idx !== nothing && original_idx in family_indices
                local_idx = findfirst(x -> x == original_idx, family_indices)
                if local_idx !== nothing
                    individual_samples = family_mcmc[:, local_idx]
                    
                    try
                        density_obj = kde(individual_samples)
                        
                        # Plot with high alpha for top individuals
                        plot!(p, density_obj.x, density_obj.density,
                             color=colors[i],
                             alpha=alpha_top,
                             linewidth=3)
                    catch e
                        continue
                    end
                end
            end
        end
        
        # Add family mean line
        family_mean = mean(family_mcmc)
        vline!(p, [family_mean],
               color=:black,
               linewidth=3,
               linestyle=:dash,
               alpha=0.8)
        
        # Calculate ranking uncertainty
        ranking_uncertainty = calculate_ranking_uncertainty(family_mcmc)
        
        # Add text annotation
        y_lims = ylims(p)
        x_lims = xlims(p)
        x_pos = x_lims[1] + 0.7 * (x_lims[2] - x_lims[1])
        
        annotate!(p, x_pos, y_lims[2] * 0.8,
                 text("Family μ: $(round(family_mean, digits=2))\nTop $top_n highlighted\nRank σ: $(round(ranking_uncertainty, digits=1))",
                      8, :left, :black))
        
        push!(plots_array, p)
    end
    
    # Combine plots
    final_plot = plot(plots_array...,
                     layout=(1, length(plots_array)),
                     size=(400*length(plots_array), 400),
                     margin=5mm,
                     plot_title="Norway Spruce MCMC EBV Uncertainty - Family Comparisons ($trait)")
    
    return final_plot
end

# Create the main uncertainty plot for selection index
println("Creating MCMC uncertainty visualization...")
uncertainty_plot = plot_norway_spruce_mcmc_uncertainty(mcmc_samples, index_mcmc, ebv_summary, 
                                                      selected_families, family_labels,
                                                      trait="Index", n_top_individuals=5)

display(uncertainty_plot)

# Save the plot
output_path = joinpath(base_path, "norway_spruce_mcmc_uncertainty_plot.png")
savefig(uncertainty_plot, output_path)
println("Saved main uncertainty plot to: $output_path")
output_path = joinpath(base_path, "norway_spruce_mcmc_uncertainty_plot.pdf")
savefig(uncertainty_plot, output_path)
println("Saved main uncertainty plot to: $output_path")
# Create plots for individual traits if available
trait_plots = []

for trait in available_traits
    if trait != "Lev17"  # Skip Lev17 if it has no variation
        println("Creating $trait uncertainty plot...")
        trait_plot = plot_norway_spruce_mcmc_uncertainty(mcmc_samples, index_mcmc, ebv_summary,
                                                        selected_families, family_labels,
                                                        trait=trait, n_top_individuals=3)
        push!(trait_plots, trait_plot)
        
        # Save individual trait plot
        trait_output_path = joinpath(base_path, "norway_spruce_$(trait)_mcmc_uncertainty.png")
        savefig(trait_plot, trait_output_path)
        println("Saved $trait plot to: $trait_output_path")
    end
end

# Create detailed ranking analysis for the highest performing family
if length(selected_families) > 0
    high_perf_family = selected_families[1]
    
    function analyze_norway_spruce_ranking_confidence(index_mcmc, ebv_summary, family_id)
        family_mask = ebv_summary.Family_ID .== family_id
        family_indices = findall(family_mask)
        family_ids = ebv_summary.Individual_ID[family_mask]
        
        if length(family_indices) == 0
            return DataFrame()
        end
        
        family_mcmc = index_mcmc[:, family_indices]
        n_samples, n_individuals = size(family_mcmc)
        
        # Calculate mean EBVs and ranking
        mean_ebvs = [mean(family_mcmc[:, i]) for i in 1:n_individuals]
        mean_ranking = sortperm(mean_ebvs, rev=true)
        
        # Calculate ranking confidence
        confidence_results = DataFrame(
            Individual_ID = family_ids,
            Mean_Index = mean_ebvs,
            Index_SD = [std(family_mcmc[:, i]) for i in 1:n_individuals],
            Mean_Rank = [findfirst(x -> x == i, mean_ranking) for i in 1:n_individuals]
        )
        
        # Calculate probabilities of being in top percentiles
        n_top10 = max(1, div(n_individuals, 10))
        n_top25 = max(1, div(n_individuals, 4))
        n_top50 = max(1, div(n_individuals, 2))
        
        confidence_results[!, :Prob_Top10] = zeros(n_individuals)
        confidence_results[!, :Prob_Top25] = zeros(n_individuals)
        confidence_results[!, :Prob_Top50] = zeros(n_individuals)
        
        for s in 1:n_samples
            sample_values = family_mcmc[s, :]
            ranks = sortperm(sample_values, rev=true)
            
            for (rank, ind_pos) in enumerate(ranks)
                if rank <= n_top10
                    confidence_results[ind_pos, :Prob_Top10] += 1.0
                end
                if rank <= n_top25
                    confidence_results[ind_pos, :Prob_Top25] += 1.0
                end
                if rank <= n_top50
                    confidence_results[ind_pos, :Prob_Top50] += 1.0
                end
            end
        end
        
        # Convert to probabilities
        confidence_results[!, :Prob_Top10] ./= n_samples
        confidence_results[!, :Prob_Top25] ./= n_samples
        confidence_results[!, :Prob_Top50] ./= n_samples
        
        # Sort by mean index
        sort!(confidence_results, :Mean_Index, rev=true)
        
        return confidence_results
    end
    
    println("Analyzing ranking confidence for high-performing family...")
    confidence_analysis = analyze_norway_spruce_ranking_confidence(index_mcmc, ebv_summary, high_perf_family)
    
    if nrow(confidence_analysis) > 0
        println("\nTop 10 individuals in highest performing family:")
        println("Individual_ID | Mean_Index | Index_SD | Mean_Rank | P(Top10%) | P(Top25%) | P(Top50%)")
        println("-" ^ 85)
        
        for i in 1:min(10, nrow(confidence_analysis))
            row = confidence_analysis[i, :]
            println("$(lpad(row.Individual_ID, 11)) | $(rpad(round(row.Mean_Index, digits=2), 10)) | $(rpad(round(row.Index_SD, digits=3), 8)) | $(lpad(row.Mean_Rank, 9)) | $(rpad(round(row.Prob_Top10, digits=3), 9)) | $(rpad(round(row.Prob_Top25, digits=3), 9)) | $(rpad(round(row.Prob_Top50, digits=3), 9))")
        end
        
        # Save detailed results
        confidence_output_path = joinpath(base_path, "norway_spruce_ranking_confidence.csv")
        CSV.write(confidence_output_path, confidence_analysis)
        println("\nSaved ranking confidence analysis to: $confidence_output_path")
    end
end

# Save EBV summary statistics
summary_output_path = joinpath(base_path, "norway_spruce_ebv_summary.csv")
CSV.write(summary_output_path, ebv_summary)
println("Saved EBV summary statistics to: $summary_output_path")

println("\n" * "="^80)
println("NORWAY SPRUCE MCMC ANALYSIS COMPLETE")
println("="^80)
println("Files generated in: $base_path")
println("  - norway_spruce_mcmc_uncertainty_plot.png: Main uncertainty visualization")
println("  - norway_spruce_[trait]_mcmc_uncertainty.png: Individual trait plots") 
println("  - norway_spruce_ranking_confidence.csv: Detailed confidence analysis")
println("  - norway_spruce_ebv_summary.csv: EBV summary statistics")
println("\nKey insights:")
println("  - Overlapping density curves show ranking uncertainty between individuals")
println("  - Thick lines highlight top performers within each family")
println("  - Family mean shown as dashed vertical line")
println("  - Ranking uncertainty (σ) quantifies selection confidence")
println("="^80)