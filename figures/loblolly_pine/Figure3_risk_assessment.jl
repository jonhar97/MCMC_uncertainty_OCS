"""
Figure 3: Risk Assessment of MAP-OCS Selections - Norway Spruce
================================================================

Recreates the original Figure 3 design:
- Top panel: Robustness scores vs rank by MAP contribution
- Bottom panel: MAP contribution values
- Both colored by risk category (quartile-based)

Author: Jon (Skogforsk)
Date: January 2026
"""

using CSV
using DataFrames
using Plots
using Statistics
using Printf

println("""
================================================================================
FIGURE 3: RISK ASSESSMENT OF MAP-OCS SELECTIONS - NORWAY SPRUCE
================================================================================
""")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input files
#ROBUSTNESS_FILE = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\Save\\robustness_with_exclusions.csv"
ROBUSTNESS_FILE =  "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData\\Save\\stable_robustness_results_926.csv"  
# Output
OUTPUT_DIR = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData\\Save"
OUTPUT_FILE = joinpath(OUTPUT_DIR, "SupplFigure3_risk_assessment_926.pdf")

println("Loading data...")


# ============================================================================
# LOAD DATA
# ============================================================================

robustness_df = CSV.read(ROBUSTNESS_FILE, DataFrame)

println("✓ Data loaded")
println("  Total individuals: $(nrow(robustness_df))")
println("  Selected in MAP-OCS: $(sum(robustness_df.selected_in_map))")

# ============================================================================
# PREPARE DATA - FILTER TO SELECTED ONLY
# ============================================================================

# Filter to only MAP-OCS selected individuals
selected_df = filter(row -> row.selected_in_map, robustness_df)

# Sort by MAP contribution (descending) and add rank
sort!(selected_df, :map_contribution, rev=true)
selected_df.rank = 1:nrow(selected_df)

println("\nAnalyzing $(nrow(selected_df)) MAP-OCS selected individuals")

# ============================================================================
# DEFINE RISK CATEGORIES BY QUARTILE
# ============================================================================

# Map quartiles to risk categories
risk_category_map = Dict(
    "Q4 (Highest Impact)" => "Low Risk (Robust)",
    "Q3 (Moderate-High)" => "Moderate Risk", 
    "Q2 (Low-Moderate)" => "Moderate Risk",
    "Q1 (Lowest Impact)" => "High Risk"
)

selected_df.risk_category = [risk_category_map[q] for q in selected_df.robustness_quartile]

# Define colors
risk_colors = Dict(
    "Low Risk (Robust)" => :green,
    "Moderate Risk" => :orange,
    "High Risk" => :red
)

# ============================================================================
# CREATE FIGURE 3: TWO-PANEL RISK ASSESSMENT
# ============================================================================

println("\nCreating Figure 3...")

# Get colors for plotting
marker_colors = [risk_colors[cat] for cat in selected_df.risk_category]

# ============================================================================
# CALCULATE SHARED POSITIONS FOR HORIZONTAL ALIGNMENT
# ============================================================================

# Calculate x-position for panel letters (same for both panels)
x_pos_letter = minimum(selected_df.rank) - (maximum(selected_df.rank) - minimum(selected_df.rank)) * 0.08

# ============================================================================
# PANEL 1 (TOP): Robustness Score vs Rank
# ============================================================================

p1 = scatter(selected_df.rank,
             selected_df.robustness_score,
             xlabel="Rank by MAP Contribution",
             ylabel="Robustness Score",
             title="",  # No title
             marker=:circle,
             markersize=8,
             markerstrokewidth=0,
             alpha=0.8,
             color=marker_colors,
             legend=false,
             dpi=300)

# Add mean line (no label, no legend)
hline!(p1, [mean(selected_df.robustness_score)],
       color=:black, linestyle=:dash, linewidth=2, 
       label="", alpha=0.6)

# Add panel letter "a" in top-left (aligned horizontally with "b")
y_pos_a = maximum(selected_df.robustness_score) * 1.08
annotate!(p1, x_pos_letter, y_pos_a, text("a", :left, 16, :bold))

# ============================================================================
# PANEL 2 (BOTTOM): MAP Contribution Values
# ============================================================================

p2 = bar(selected_df.rank,
         selected_df.map_contribution,
         xlabel="Rank by MAP Contribution", 
         ylabel="MAP Contribution",
         title="",  # No title
         legend=false,
         color=marker_colors,
         alpha=0.7,
         linewidth=0,
         dpi=300)

# Add panel letter "b" in top-left (aligned horizontally with "a")
y_pos_b = maximum(selected_df.map_contribution) * 1.08
annotate!(p2, x_pos_letter, y_pos_b, text("b", :left, 16, :bold))

# ============================================================================
# COMBINE PANELS
# ============================================================================

p = plot(p1, p2, layout=(2, 1), size=(1000, 800), margin=8Plots.mm)

# ============================================================================
# SAVE FIGURE
# ============================================================================

savefig(p, OUTPUT_FILE)
println("\n✓ Figure saved: $OUTPUT_FILE")

# Also save PNG
output_png = replace(OUTPUT_FILE, ".pdf" => ".png")
savefig(p, output_png)
println("✓ Figure saved: $output_png")

# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================

println("\n" * "="^80)
println("RISK ASSESSMENT SUMMARY")
println("="^80)
println()

# Statistics by risk category
risk_stats = combine(groupby(selected_df, :risk_category)) do subdf
    DataFrame(
        n = nrow(subdf),
        mean_robustness = mean(subdf.robustness_score),
        std_robustness = std(subdf.robustness_score),
        mean_contribution = mean(subdf.map_contribution),
        median_rank = median(subdf.rank)
    )
end

println(@sprintf("%-20s %6s %15s %15s %10s", 
                 "Risk Category", "n", "Mean Robustness", "Mean Contrib", "Med. Rank"))
println("-"^80)

for row in eachrow(risk_stats)
    println(@sprintf("%-20s %6d %10.4f±%.4f %10.6f %10.0f",
                    row.risk_category,
                    row.n,
                    row.mean_robustness,
                    row.std_robustness,
                    row.mean_contribution,
                    row.median_rank))
end

println("\n" * "="^80)
println("FIGURE 3 COMPLETE!")
println("="^80)
println("\nThis figure shows:")
println("  - Top panel: Robustness scores vs rank by MAP contribution")
println("  - Bottom panel: MAP contribution values")
println("  - Colors indicate risk categories based on robustness quartiles")
println()
println("Key finding:")
println("  High-risk individuals (Q1, red) have low robustness scores")
println("  Low-risk individuals (Q4, green) have high robustness scores")
println("  → Clear risk stratification among selected individuals")
println("="^80)
