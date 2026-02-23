"""
Publication-Ready Figure 1: Within-Family Posterior Uncertainty
================================================================

Creates kernel density plots showing posterior distributions of breeding values
for individuals within three representative families (high, medium, low performing).

Figure uses panel labels (a, b, c) instead of titles for publication.
"""

using Plots
using KernelDensity
using Statistics
using Printf
using DataFrames
using CSV
using DelimitedFiles
using Colors  # For RGB color specification

# ============================================================================
# CONFIGURATION
# ============================================================================

# Update these paths to match your data location
DATA_PATH = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData"
RESULTS_PATH = joinpath(DATA_PATH, "results_G_926")
SAVE_PATH = joinpath(DATA_PATH, "Save")

# MCMC samples files - one per trait (1218 individuals × 1000 iterations)
# Each row = individual, each column = MCMC iteration
MCMC_HT6_FILE = joinpath(RESULTS_PATH, "MCMC_samples_EBV_HT6.txt")
MCMC_DBH6_FILE = joinpath(RESULTS_PATH, "MCMC_samples_EBV_DBH6.txt")
MCMC_GV6_FILE = joinpath(RESULTS_PATH, "MCMC_samples_EBV_GV6.txt")
MCMC_WDN4_FILE = joinpath(RESULTS_PATH, "MCMC_samples_EBV_WDN4.txt")  # wood density

# Selection index formula: I = (Hjd17 + Htv17 - Sprant17) / 3
INDEX_FORMULA = (ht, dbh, gv, wdn) -> (ht .+ dbh .- gv + wdn) ./ 4

# JWAS phenotypes file with family information
PHENOTYPES_FILE = joinpath(DATA_PATH, "JWAS_phenotypes_926_index_with_family.txt")
FAMILY_COLUMN = "Family"  # Column name in phenotypes file

# Selected families to display
# Run identify_families_helper.jl first to get recommendations
# These are placeholder values - update after running helper script
SELECTED_FAMILIES = ["142024_x_44116", "52010_x_52004", "202096_x_22022"]  # High, Medium, Low performing
FAMILY_LABELS = ["High-performing", "Medium-performing", "Low-performing"]

# Number of top individuals to highlight per family
N_TOP_HIGHLIGHT = 5

# Output
OUTPUT_FILE = joinpath(SAVE_PATH, "Figure1_publication_taeda.pdf")

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

"""
Load MCMC samples for one trait
Returns: Matrix (individuals × iterations)
"""
function load_trait_mcmc(filepath::String, trait_name::String)
    println("  Loading $trait_name from: $(basename(filepath))")
    
    if !isfile(filepath)
        error("MCMC samples file not found: $filepath")
    end
    
    # Load as matrix, skipping the first row (header with iteration numbers or column names)
    # File format: rows = MCMC iterations, columns = individuals
    # We need to transpose to get: rows = individuals, columns = iterations
    data_raw = readdlm(filepath, ',', Float64, '\n', skipstart=1)
    
    # Transpose: iterations × individuals → individuals × iterations
    data = transpose(data_raw)
    
    n_individuals = size(data, 1)
    n_iterations = size(data, 2)
    
    println("    ✓ Loaded: $n_individuals individuals × $n_iterations MCMC iterations")
    
    return data
end

"""
Load all three traits and compute selection index
Returns: DataFrame with selection index values (individuals × iterations)
"""
function load_selection_index_mcmc(ht_file::String, dbh_file::String, 
                                   gv_file::String, wdn_file::String, index_formula)
    println("\nLoading MCMC samples for selection index I = (HT6 + DBH6 - GV6 + WDN4) / 4")
    println("="^80)
    
    # Load each trait
    ht_samples = load_trait_mcmc(ht_file, "HTV6")
    dbh_samples = load_trait_mcmc(dbh_file, "DBH6")
    gv_samples = load_trait_mcmc(gv_file, "GV6")
    wdn_samples = load_trait_mcmc(wdn_file, "WDN4")
    
    # Verify dimensions match
    if size(ht_samples) != size(dbh_samples) || size(gv_samples) != size(wdn_samples)
        error("Dimension mismatch between trait files!")
    end
    
    n_individuals = size(ht_samples, 1)
    n_iterations = size(ht_samples, 2)
    
    println("\n  Computing selection index...")
    
    # Compute selection index for each individual × iteration
    index_samples = index_formula(ht_samples, dbh_samples, gv_samples, wdn_samples)
    
    println("    ✓ Selection index computed: $n_individuals individuals × $n_iterations iterations")
    
    # Convert to DataFrame for consistency with rest of code
    df = DataFrame(index_samples, :auto)
    
    # Also return the raw trait data in case we need it
    trait_data = Dict(
        "HT6" => ht_samples,
        "DBH6" => dbh_samples,
        "GV6" => gv_samples,
        "WDN4" => wdn_samples,
        "Index" => index_samples
    )
    
    return df, trait_data
end

"""
Load family IDs from JWAS phenotypes file
Returns: Vector of family IDs (length = n_individuals)
"""
function load_family_ids(phenotypes_file::String, family_column::String)
    println("\nLoading family information from: $(basename(phenotypes_file))")
    println("="^80)
    
    if !isfile(phenotypes_file)
        error("Phenotypes file not found: $phenotypes_file")
    end
    
    # Load phenotypes file as DataFrame
    # Try CSV first, fall back to delimited text if needed
    phenotypes = try
        CSV.read(phenotypes_file, DataFrame)
    catch
        # If CSV fails, try space/tab delimited
        CSV.read(phenotypes_file, DataFrame, delim=' ', ignorerepeated=true)
    end
    
    println("  ✓ Loaded phenotypes: $(nrow(phenotypes)) individuals")
    println("  Available columns: $(names(phenotypes))")
    
    # Check if family column exists
    if !(family_column in names(phenotypes))
        error("Family column '$family_column' not found. Available columns: $(names(phenotypes))")
    end
    
    # Extract family IDs
    family_ids = phenotypes[!, family_column]
    
    # Convert to regular Vector if it's a PooledVector
    family_ids = Vector{String}(family_ids)
    
    println("  ✓ Extracted family IDs from column: $family_column")
    println("    Unique families: $(length(unique(family_ids)))")
    
    return family_ids
end

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

"""
Calculate posterior mean breeding values for each individual
"""
function calculate_posterior_means(mcmc_samples::DataFrame)
    n_individuals = nrow(mcmc_samples)
    posterior_means = zeros(n_individuals)
    
    for i in 1:n_individuals
        # Get all MCMC samples for this individual (as a row)
        individual_samples = Vector(mcmc_samples[i, :])
        posterior_means[i] = mean(individual_samples)
    end
    
    return posterior_means
end

"""
Get individuals within a specific family
"""
function get_family_individuals(family_ids::Vector, target_family::String)
    return findall(family_ids .== target_family)
end

"""
Identify top N individuals within a family by posterior mean
"""
function get_top_individuals(family_indices::Vector{Int}, 
                            posterior_means::Vector{Float64}, 
                            n_top::Int)
    
    family_means = posterior_means[family_indices]
    
    # Sort within family by posterior mean (descending)
    sorted_indices = sortperm(family_means, rev=true)
    
    # Get top N (or all if family is smaller)
    n_select = min(n_top, length(sorted_indices))
    top_within_family = sorted_indices[1:n_select]
    
    # Return global indices
    return family_indices[top_within_family]
end

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

"""
Create kernel density estimate for MCMC samples
"""
function create_kde(samples::Vector{Float64}; bandwidth::Float64=0.1)
    # Use KernelDensity.jl
    kde_result = kde(samples, bandwidth=bandwidth)
    return kde_result.x, kde_result.density
end

"""
Plot posterior distributions for one family (one panel)
"""
function plot_family_panel!(p, mcmc_samples::DataFrame, family_ids::Vector,
                           family_id::String, family_label::String,
                           posterior_means::Vector{Float64},
                           n_top::Int, panel_letter::String)
    
    # Get individuals in this family
    family_indices = get_family_individuals(family_ids, family_id)
    n_family = length(family_indices)
    
    println("  Processing Family $family_id ($family_label): $n_family individuals")
    
    if n_family == 0
        println("    ⚠ Warning: No individuals found in family $family_id")
        return
    end
    
    # Get top individuals
    top_indices = get_top_individuals(family_indices, posterior_means, n_top)
    
    # Calculate family statistics
    family_mean = mean(posterior_means[family_indices])
    family_std = std(posterior_means[family_indices])
    
    # Calculate mean MCMC uncertainty across family
    individual_sds = Float64[]
    for idx in family_indices
        samples = Vector(mcmc_samples[idx, :])
        push!(individual_sds, std(samples))
    end
    mean_mcmc_sd = mean(individual_sds)
    
    # Determine x-axis limits based on all family members
    all_family_samples = Float64[]
    for idx in family_indices
        samples = Vector(mcmc_samples[idx, :])
        append!(all_family_samples, samples)
    end
    x_min = minimum(all_family_samples) - 0.5
    x_max = maximum(all_family_samples) + 0.5
    
    # Estimate reasonable bandwidth
    bandwidth = 0.15 * std(all_family_samples)
    
    # Plot all individuals in family (thin gray lines)
    for idx in family_indices
        samples = Vector(mcmc_samples[idx, :])
        
        # Create KDE
        x_vals, density = create_kde(samples, bandwidth=bandwidth)
        
        # Normalize density for better visualization
        density_norm = density ./ maximum(density)
        
        # Plot thin line (gray) for all individuals
        plot!(p, x_vals, density_norm,
              color=:gray, alpha=0.3, linewidth=0.8,
              label="", xlims=(x_min, x_max))
    end
    
    # Highlight top N individuals (thick colored lines) - ORIGINAL color scheme from your figure
    # Beautiful muted/pastel colors matching your original Figure 1
    colors_top = [
        RGB(0.12, 0.47, 0.71),  # Muted blue
        RGB(1.0, 0.50, 0.05),   # Orange
        RGB(0.17, 0.63, 0.17),  # Green  
        RGB(0.84, 0.15, 0.16),  # Red
        RGB(0.58, 0.40, 0.74)   # Purple
    ]
    
    for (i, idx) in enumerate(top_indices)
        samples = Vector(mcmc_samples[idx, :])
        
        # Create KDE
        x_vals, density = create_kde(samples, bandwidth=bandwidth)
        density_norm = density ./ maximum(density)
        
        # Plot thick line with original colors
        color_idx = mod1(i, length(colors_top))
        plot!(p, x_vals, density_norm,
              color=colors_top[color_idx], alpha=0.9, linewidth=2.5,
              label="")
    end
    
    # Add family mean (dashed vertical line)
    vline!(p, [family_mean],
           color=:black, linestyle=:dash, linewidth=2,
           label="")
    
    # Set axis labels and remove grid
    plot!(p, xlabel="Breeding value",
          ylabel="Normalized density",
          guidefontsize=11,
          tickfontsize=10,
          legend=false,
          framestyle=:box,
          grid=false)  # Remove grid
    
    # Add panel label in top-left corner (bigger font)
    annotate!(p, x_min + (x_max - x_min) * 0.05,
              0.95,
              text(panel_letter, :left, 16, :bold))  # Increased from 14 to 16
    
    # Add family info and statistics in top-right (INCREASED text size from 8 to 10)
    stats_text = "$family_label\n" *
                 "(Family $family_id, n=$n_family)\n" *
                 "μ = $(round(family_mean, digits=2))\n" *
                 "σ = $(round(family_std, digits=2))\n" *
                 "MCMC SD = $(round(mean_mcmc_sd, digits=2))"
    
    annotate!(p, x_max - (x_max - x_min) * 0.05,
              0.82,
              text(stats_text, :right, 10))  # Increased from 8 to 10
end

"""
Create complete three-panel figure
"""
function create_figure1(mcmc_samples::DataFrame, family_ids::Vector,
                       selected_families::Vector{String}, 
                       family_labels::Vector{String},
                       n_top::Int)
    
    println("\n" * "="^80)
    println("CREATING FIGURE 1: WITHIN-FAMILY POSTERIOR UNCERTAINTY")
    println("="^80)
    
    # Calculate posterior means for all individuals
    println("\nCalculating posterior means...")
    posterior_means = calculate_posterior_means(mcmc_samples)
    println("  ✓ Posterior means calculated")
    
    # Create three-panel plot
    println("\nCreating three-panel figure...")
    p = plot(layout=(1, 3), size=(1800, 500),
             left_margin=10Plots.mm, bottom_margin=8Plots.mm,
             right_margin=5Plots.mm, top_margin=5Plots.mm)
    
    # Panel letters
    panel_letters = ["a", "b", "c"]
    
    # Plot each family
    for (i, (fam_id, fam_label)) in enumerate(zip(selected_families, family_labels))
        plot_family_panel!(p[i], mcmc_samples, family_ids,
                          fam_id, fam_label,
                          posterior_means, n_top, panel_letters[i])
    end
    
    println("\n✓ Figure created successfully!")
    
    return p
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

"""
Main function - creates publication-ready Figure 1
"""
function main()
    println("="^80)
    println("FIGURE 1: WITHIN-FAMILY POSTERIOR UNCERTAINTY - NORWAY SPRUCE")
    println("Publication-ready version with panel labels (a, b, c)")
    #println("Selection Index: I = (Hjd17 + Htv17 - Sprant17) / 3")
    println("="^80)
    
    # Load MCMC samples for all traits and compute selection index
    mcmc_samples, trait_data = load_selection_index_mcmc(
        MCMC_HT6_FILE,
        MCMC_DBH6_FILE,
        MCMC_GV6_FILE,
        MCMC_WDN4_FILE,
        INDEX_FORMULA
    )
    
    # Load family IDs from phenotypes file
    family_ids = load_family_ids(PHENOTYPES_FILE, FAMILY_COLUMN)
    
    # Validate
    if nrow(mcmc_samples) != length(family_ids)
        error("Data mismatch: $(nrow(mcmc_samples)) individuals in MCMC samples vs $(length(family_ids)) in phenotypes file")
    end
    
    println("\n" * "="^80)
    println("DATA VALIDATION")
    println("="^80)
    println("  ✓ Total individuals: $(nrow(mcmc_samples))")
    println("  ✓ MCMC iterations: $(ncol(mcmc_samples))")
    println("  ✓ Unique families: $(length(unique(family_ids)))")
    
    # Check if selected families exist
    for (fam_id, fam_label) in zip(SELECTED_FAMILIES, FAMILY_LABELS)
        n_in_family = sum(family_ids .== fam_id)
        if n_in_family == 0
            @warn "Selected family $fam_id ($fam_label) has 0 individuals!"
            println("    Available families: $(sort(unique(family_ids)))")
        else
            println("  ✓ Family $fam_id ($fam_label): $n_in_family individuals")
        end
    end
    
    # Create figure
    println("\n" * "="^80)
    println("CREATING FIGURE")
    println("="^80)
    
    fig = create_figure1(mcmc_samples, family_ids, 
                        SELECTED_FAMILIES, FAMILY_LABELS,
                        N_TOP_HIGHLIGHT)
    
    # Save
    savefig(fig, OUTPUT_FILE)
    println("\n✓ Figure saved to: $OUTPUT_FILE")
    
    # Also save PNG version
    png_file = replace(OUTPUT_FILE, ".pdf" => ".png")
    savefig(fig, png_file)
    println("✓ PNG version saved to: $png_file")
    
    println("\n" * "="^80)
    println("COMPLETE!")
    println("="^80)
    println("\nNext steps:")
    println("1. Review the figure to ensure families look correct")
    println("2. Adjust SELECTED_FAMILIES if needed to show representative examples")
    println("3. Update n values in figure caption if different from displayed")
    
    # Return both figure and data for inspection
    return Dict(
        "figure" => fig,
        "mcmc_samples" => mcmc_samples,
        "family_ids" => family_ids,
        "trait_data" => trait_data
    )
end

# ============================================================================
# QUICK START GUIDE
# ============================================================================

println("""
================================================================================
FIGURE 1 - WITHIN-FAMILY POSTERIOR UNCERTAINTY
Publication-ready with panel labels (a, b, c)
================================================================================



USAGE:
1. Verify your pedigree file path is correct
2. Update SELECTED_FAMILIES if you want different examples
3. Run: include("Figure1_within_family_uncertainty_taeda.jl")
        results = main()

The figure will have:
✓ Three panels (a, b, c) for high/medium/low performing families
✓ Thin gray curves for all individuals within each family
✓ Thick colored curves for top 5 individuals per family (by posterior mean)
✓ Dashed vertical lines indicating family means
✓ Clean, publication-ready formatting (no titles, just panel letters)

OUTPUT FILES:
✓ Figure1_publication_taeda.pdf (vector graphics)
✓ Figure1_publication_taeda.png (raster for preview)

FIGURE CAPTION:
"Posterior uncertainty in estimated breeding values reveals ranking ambiguity 
within Loblolly pine families. Kernel density plots show the posterior 
distributions of selection index breeding values (HT6 + DBH6 - GV6 + WDN4) 
for individuals within three representative families: high-performing, 
medium-performing, and low-performing (family IDs and sample sizes shown in 
panel titles). Each thin curve represents the posterior distribution for one 
individual based on 1000 MCMC samples. Thick curves highlight the top 5 
individuals within each family ranked by posterior mean. Dashed vertical 
lines indicate family means. Panel labels (a, b, c) denote performance level."

NOTES:
- Selection index: I = (HT6 + DBH6 - GV6 + WDN4) / 4
  where HT6 = height at age 6, DBH6 = diameter at age 6,
  GV6 = spiral grain at age 6, WDN4 = wood density at age 4
- Family IDs are in format "Mom_x_Dad" (e.g., "142024_x_44116")
- Update family IDs and sample sizes in caption after running the script

TROUBLESHOOTING:
- If pedigree file not found: Update PEDIGREE_FILE path
- If families have 0 individuals: Set EXTRACT_FAMILY_FROM_IDS = true
- If family IDs wrong: Check sire column in pedigree (might be column 3)
- To see family sizes before plotting: 
    using DelimitedFiles
    ped = readdlm(PEDIGREE_FILE)
    family_counts = countmap(ped[:, 2])  # Adjust column if needed

================================================================================
""")

# Uncomment to run:
# results = main()
