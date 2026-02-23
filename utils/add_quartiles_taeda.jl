"""
Add Robustness Quartiles to Loblolly Pine Data
==============================================

Reads the stable_robustness_results_926.csv file, assigns quartiles based on
robustness scores, and saves with the new column.

Author: Jon (Skogforsk)
Date: January 2026
"""

using CSV
using DataFrames
using Statistics

println("""
================================================================================
ADDING ROBUSTNESS QUARTILES - LOBLOLLY PINE
================================================================================
""")

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData\\Save\\stable_robustness_results_926.csv"
OUTPUT_FILE = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData\\Save\\robustness_with_quartiles_taeda.csv"

println("Loading data...")

# ============================================================================
# LOAD DATA
# ============================================================================

df = CSV.read(INPUT_FILE, DataFrame)

println("✓ Data loaded")
println("  Total individuals: $(nrow(df))")
println("  Columns: $(names(df))")

# ============================================================================
# ASSIGN QUARTILES
# ============================================================================

println("\nAssigning quartiles based on robustness scores...")

# Calculate quartile thresholds
sorted_scores = sort(df.robustness_score)
n = length(sorted_scores)

# Quartile boundaries (using standard definition)
q1_threshold = sorted_scores[Int(ceil(n * 0.25))]
q2_threshold = sorted_scores[Int(ceil(n * 0.50))]
q3_threshold = sorted_scores[Int(ceil(n * 0.75))]

println("  Q1 threshold (25th percentile): $(round(q1_threshold, digits=6))")
println("  Q2 threshold (50th percentile): $(round(q2_threshold, digits=6))")
println("  Q3 threshold (75th percentile): $(round(q3_threshold, digits=6))")

# Assign quartiles
df.robustness_quartile = Vector{String}(undef, nrow(df))

for i in 1:nrow(df)
    score = df.robustness_score[i]
    if score <= q1_threshold
        df.robustness_quartile[i] = "Q1 (Lowest Impact)"
    elseif score <= q2_threshold
        df.robustness_quartile[i] = "Q2 (Low-Moderate)"
    elseif score <= q3_threshold
        df.robustness_quartile[i] = "Q3 (Moderate-High)"
    else
        df.robustness_quartile[i] = "Q4 (Highest Impact)"
    end
end

# ============================================================================
# VERIFY QUARTILES
# ============================================================================

println("\nQuartile distribution:")
quartile_counts = combine(groupby(df, :robustness_quartile), nrow => :count)
sort!(quartile_counts, :robustness_quartile, 
      by = x -> something(findfirst(==(x), ["Q1 (Lowest Impact)", "Q2 (Low-Moderate)", 
                                             "Q3 (Moderate-High)", "Q4 (Highest Impact)"]), 999))

for row in eachrow(quartile_counts)
    pct = (row.count / nrow(df)) * 100
    println("  $(row.robustness_quartile): $(row.count) ($(round(pct, digits=1))%)")
end

# Show statistics by quartile
println("\nRobustness score statistics by quartile:")
quartile_stats = combine(groupby(df, :robustness_quartile)) do subdf
    DataFrame(
        n = nrow(subdf),
        mean = mean(subdf.robustness_score),
        min = minimum(subdf.robustness_score),
        max = maximum(subdf.robustness_score)
    )
end

sort!(quartile_stats, :robustness_quartile,
      by = x -> something(findfirst(==(x), ["Q1 (Lowest Impact)", "Q2 (Low-Moderate)", 
                                             "Q3 (Moderate-High)", "Q4 (Highest Impact)"]), 999))

using Printf
println(@sprintf("%-25s %5s %10s %10s %10s", "Quartile", "n", "Mean", "Min", "Max"))
println("-"^80)
for row in eachrow(quartile_stats)
    println(@sprintf("%-25s %5d %10.6f %10.6f %10.6f",
                    row.robustness_quartile, row.n, row.mean, row.min, row.max))
end

# ============================================================================
# SAVE OUTPUT
# ============================================================================

println("\nSaving data with quartiles...")
CSV.write(OUTPUT_FILE, df)

println("✓ Data saved: $OUTPUT_FILE")
println("  Total rows: $(nrow(df))")
println("  Columns: $(join(names(df), ", "))")

println("\n" * "="^80)
println("QUARTILE ASSIGNMENT COMPLETE!")
println("="^80)
println("\nYou can now use this file for Figure 3:")
println("  ROBUSTNESS_FILE = \"$OUTPUT_FILE\"")
println("="^80)
