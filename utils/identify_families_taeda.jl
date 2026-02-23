"""
Helper Script: Identify Families for Figure 1
==============================================

Analyzes family structure and breeding values to recommend which families
to display in Figure 1 (within-family EBV posterior variation).

Selection criteria:
- High-performing: High mean BV, good family size
- Medium-performing: Average mean BV, good family size
- Low-performing: Low mean BV, good family size

Author: Jon (Skogforsk)
Date: January 2026
"""

using CSV
using DataFrames
using Statistics
using Printf

println("""
================================================================================
IDENTIFY FAMILIES FOR FIGURE 1 - LOBLOLLY PINE
================================================================================
""")

# ============================================================================
# CONFIGURATION
# ============================================================================

PHENOTYPES_FILE = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData\\JWAS_phenotypes_926_index_with_family.txt"
MCMC_DIR = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData\\results_G_926"
FAMILY_COLUMN = "Family"

# Minimum family size to consider (need enough individuals for visualization)
MIN_FAMILY_SIZE = 5

println("Loading data...")

# ============================================================================
# LOAD PHENOTYPE DATA
# ============================================================================

pheno_df = CSV.read(PHENOTYPES_FILE, DataFrame)

println("✓ Phenotype data loaded")
println("  Total individuals: $(nrow(pheno_df))")

# ============================================================================
# LOAD MCMC BREEDING VALUES FOR ALL TRAITS
# ============================================================================

println("\nLoading MCMC breeding values for all traits...")

# File paths
MCMC_HT6_FILE = joinpath(MCMC_DIR, "MCMC_samples_EBV_HT6.txt")
MCMC_DBH6_FILE = joinpath(MCMC_DIR, "MCMC_samples_EBV_DBH6.txt")
MCMC_GV6_FILE = joinpath(MCMC_DIR, "MCMC_samples_EBV_GV6.txt")
MCMC_WDN4_FILE = joinpath(MCMC_DIR, "MCMC_samples_EBV_WDN4.txt")

# Load each trait
println("  Loading HT6...")
ht_data = CSV.read(MCMC_HT6_FILE, DataFrame, header=false)
println("  Loading DBH6...")
dbh_data = CSV.read(MCMC_DBH6_FILE, DataFrame, header=false)
println("  Loading GV6...")
gv_data = CSV.read(MCMC_GV6_FILE, DataFrame, header=false)
println("  Loading WDN4...")
wdn_data = CSV.read(MCMC_WDN4_FILE, DataFrame, header=false)

println("✓ All traits loaded")
println("  Dimensions: $(size(ht_data))")

# Extract IDs and MCMC samples
# File format: rows = MCMC iterations, columns = individuals
# First row contains individual IDs, remaining rows are MCMC samples
ids = Vector(ht_data[1, :])  # First row = IDs
ht_samples = Matrix(ht_data[2:end, :])  # Remaining rows = MCMC samples
dbh_samples = Matrix(dbh_data[2:end, :])
gv_samples = Matrix(gv_data[2:end, :])
wdn_samples = Matrix(wdn_data[2:end, :])

println("  Individuals: $(length(ids))")
println("  MCMC iterations: $(size(ht_samples, 1))")

# ============================================================================
# CALCULATE SELECTION INDEX
# ============================================================================

println("\nCalculating selection index...")

# Selection index formula: I = (HT6 + DBH6 - GV6 + WDN4) / 4
INDEX_FORMULA = (ht, dbh, gv, wdn) -> (ht .+ dbh .- gv .+ wdn) ./ 4

# Calculate index for each MCMC iteration
index_samples = INDEX_FORMULA(ht_samples, dbh_samples, gv_samples, wdn_samples)

println("✓ Selection index calculated")
println("  Index dimensions: $(size(index_samples))")

# Use index samples for downstream analysis
mcmc_samples = index_samples

# Calculate mean and std across MCMC samples for each individual
# Rows = MCMC iterations, columns = individuals
mean_bv = vec(mean(mcmc_samples, dims=1))  # dims=1 to average over rows (iterations)
std_bv = vec(std(mcmc_samples, dims=1))

# Create DataFrame with BV statistics
bv_stats = DataFrame(
    ID = ids,
    mean_bv = mean_bv,
    std_bv = std_bv,
    cv = std_bv ./ abs.(mean_bv)  # Coefficient of variation
)

# ============================================================================
# MERGE WITH PHENOTYPES TO GET FAMILY INFO
# ============================================================================

println("\nMerging with family information...")

# Merge
merged_df = leftjoin(bv_stats, pheno_df, on=:ID)

println("✓ Data merged")
println("  Individuals with family info: $(sum(.!ismissing.(merged_df[:, FAMILY_COLUMN])))")

# ============================================================================
# ANALYZE FAMILIES
# ============================================================================

println("\nAnalyzing family structure and performance...")

# Calculate family-level statistics
family_stats = combine(groupby(merged_df, FAMILY_COLUMN)) do subdf
    DataFrame(
        n = nrow(subdf),
        mean_bv = mean(subdf.mean_bv),
        std_bv_within = std(subdf.mean_bv),  # Variation of mean BVs within family
        mean_std_bv = mean(subdf.std_bv),    # Mean of individual posterior SDs
        min_bv = minimum(subdf.mean_bv),
        max_bv = maximum(subdf.mean_bv),
        range_bv = maximum(subdf.mean_bv) - minimum(subdf.mean_bv)
    )
end

# Filter families with sufficient size
family_stats_filtered = filter(row -> row.n >= MIN_FAMILY_SIZE, family_stats)

println("  Total families: $(nrow(family_stats))")
println("  Families with ≥$(MIN_FAMILY_SIZE) offspring: $(nrow(family_stats_filtered))")

if nrow(family_stats_filtered) == 0
    error("No families with at least $MIN_FAMILY_SIZE offspring. Lower MIN_FAMILY_SIZE?")
end

# ============================================================================
# RANK FAMILIES BY PERFORMANCE
# ============================================================================

println("\nRanking families by mean breeding value...")

# Sort by mean BV
sort!(family_stats_filtered, :mean_bv, rev=true)

# Add percentile ranks
family_stats_filtered.percentile = (1:nrow(family_stats_filtered)) ./ nrow(family_stats_filtered) .* 100

# Identify candidates for high, medium, low
n_families = nrow(family_stats_filtered)

# High: Top 10%
high_candidates = family_stats_filtered[1:max(1, Int(ceil(n_families * 0.1))), :]
# Medium: Around 45-55%
medium_start = Int(floor(n_families * 0.45))
medium_end = Int(ceil(n_families * 0.55))
medium_candidates = family_stats_filtered[medium_start:medium_end, :]
# Low: Bottom 10%
low_candidates = family_stats_filtered[max(1, n_families - Int(ceil(n_families * 0.1))):end, :]

# ============================================================================
# SELECT BEST REPRESENTATIVES
# ============================================================================

println("\nSelecting representative families...")

# For each category, prefer families with:
# 1. Good family size (not too small, not too large)
# 2. Good within-family variation (shows uncertainty)
# 3. Representative of the category

function select_best_family(candidates, target_size=8)
    # Score families
    candidates.score = abs.(candidates.n .- target_size) .+ 
                      (1.0 .- candidates.std_bv_within ./ maximum(candidates.std_bv_within))
    
    sort!(candidates, :score)
    return candidates[1, FAMILY_COLUMN]
end

high_family = select_best_family(high_candidates)
medium_family = select_best_family(medium_candidates)
low_family = select_best_family(low_candidates)

# Get their stats
high_stats = family_stats_filtered[family_stats_filtered[:, FAMILY_COLUMN] .== high_family, :][1, :]
medium_stats = family_stats_filtered[family_stats_filtered[:, FAMILY_COLUMN] .== medium_family, :][1, :]
low_stats = family_stats_filtered[family_stats_filtered[:, FAMILY_COLUMN] .== low_family, :][1, :]

# ============================================================================
# DISPLAY RECOMMENDATIONS
# ============================================================================

println("\n" * "="^80)
println("RECOMMENDED FAMILIES FOR FIGURE 1")
println("="^80)

function print_family_info(label, family, stats)
    println("\n$label Family:")
    println("  Family ID: $family")
    println("  Offspring: $(stats.n)")
    println("  Mean BV: $(round(stats.mean_bv, digits=4))")
    println("  Within-family SD: $(round(stats.std_bv_within, digits=4))")
    println("  BV range: [$(round(stats.min_bv, digits=4)), $(round(stats.max_bv, digits=4))]")
    println("  Mean posterior SD: $(round(stats.mean_std_bv, digits=4))")
end

print_family_info("HIGH-PERFORMING", high_family, high_stats)
print_family_info("MEDIUM-PERFORMING", medium_family, medium_stats)
print_family_info("LOW-PERFORMING", low_family, low_stats)

# ============================================================================
# GENERATE CODE FOR FIGURE 1 SCRIPT
# ============================================================================

println("\n" * "="^80)
println("CODE TO PASTE INTO FIGURE 1 SCRIPT")
println("="^80)

println("\n# Selected families to display")
println("SELECTED_FAMILIES = [")
println("    \"$high_family\",    # High-performing")
println("    \"$medium_family\",  # Medium-performing")
println("    \"$low_family\"      # Low-performing")
println("]")
println("FAMILY_LABELS = [\"High-performing\", \"Medium-performing\", \"Low-performing\"]")

# ============================================================================
# SHOW ALTERNATIVES
# ============================================================================

println("\n" * "="^80)
println("ALTERNATIVE CANDIDATES")
println("="^80)

println("\nTop 5 high-performing families:")
println(@sprintf("%-20s %5s %10s %10s %10s", "Family", "n", "Mean BV", "Within SD", "Rank"))
println("-"^80)
for i in 1:min(5, nrow(high_candidates))
    row = high_candidates[i, :]
    println(@sprintf("%-20s %5d %10.4f %10.4f %10.1f%%",
                    row[FAMILY_COLUMN], row.n, row.mean_bv, row.std_bv_within, row.percentile))
end

println("\nTop 5 medium-performing families:")
println(@sprintf("%-20s %5s %10s %10s %10s", "Family", "n", "Mean BV", "Within SD", "Rank"))
println("-"^80)
for i in 1:min(5, nrow(medium_candidates))
    row = medium_candidates[i, :]
    println(@sprintf("%-20s %5d %10.4f %10.4f %10.1f%%",
                    row[FAMILY_COLUMN], row.n, row.mean_bv, row.std_bv_within, row.percentile))
end

println("\nTop 5 low-performing families:")
println(@sprintf("%-20s %5s %10s %10s %10s", "Family", "n", "Mean BV", "Within SD", "Rank"))
println("-"^80)
for i in 1:min(5, nrow(low_candidates))
    row = low_candidates[i, :]
    println(@sprintf("%-20s %5d %10.4f %10.4f %10.1f%%",
                    row[FAMILY_COLUMN], row.n, row.mean_bv, row.std_bv_within, row.percentile))
end

# ============================================================================
# SAVE FULL FAMILY STATS
# ============================================================================

output_file = joinpath(MCMC_DIR, "family_statistics_for_figure1.csv")
CSV.write(output_file, family_stats_filtered)
println("\n✓ Full family statistics saved: $output_file")

println("\n" * "="^80)
println("FAMILY IDENTIFICATION COMPLETE!")
println("="^80)
println("\nNext step: Copy the recommended families into your Figure 1 script")
println("="^80)
