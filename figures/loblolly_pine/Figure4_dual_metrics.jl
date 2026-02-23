"""
Comprehensive Dual Metrics Visualization - Loblolly Pine
========================================================

Creates publication-quality multi-panel figure showing:
- Panel A: Individual robustness scores (bar plot)
- Panel B: Individual robustness by quartile (scatter)
- Panel C: Portfolio vulnerability metrics comparison (bar plot)
- Panel D: Contribution distribution comparison (violin plots)

Author: Jon (Skogforsk)
Date: January 2026
"""

using CSV
using DataFrames
using Plots
using StatsPlots
using Statistics
using Printf

println("""
================================================================================
COMPREHENSIVE DUAL METRICS VISUALIZATION - LOBLOLLY PINE
================================================================================
""")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input files
#ROBUSTNESS_FILE = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData\\Save\\stable_robustness_results_926.csv"
ROBUSTNESS_FILE =  "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData\\Save\\stable_robustness_results_926.csv"

SOLUTIONS_FILE = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData\\dual_metrics_taeda_results\\taeda_ocs_solutions.csv"

# Output directory
OUTPUT_DIR = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData\\dual_metrics_taeda_results"
mkpath(OUTPUT_DIR)

println("Loading data...")

# ============================================================================
# LOAD DATA
# ============================================================================

robustness_df = CSV.read(ROBUSTNESS_FILE, DataFrame)
solutions_df = CSV.read(SOLUTIONS_FILE, DataFrame)

println("✓ Data loaded")
println("  Robustness scores: $(nrow(robustness_df)) individuals")
println("  OCS solutions: $(nrow(solutions_df)) individuals")

# ============================================================================
# CALCULATE PORTFOLIO METRICS
# ============================================================================

println("\nCalculating portfolio vulnerability metrics...")

threshold = 1e-4

# MAP-OCS metrics
map_selected = solutions_df.map_contribution .> threshold
map_contributions = solutions_df.map_contribution[map_selected] .* solutions_df.standardized_bv[map_selected]

map_mean = mean(map_contributions)
map_max = maximum(map_contributions)
map_conc = sum(sort(map_contributions, rev=true)[1:min(Int(ceil(length(map_contributions)*0.1)), length(map_contributions))]) / sum(map_contributions) * 100

# Gini coefficient for MAP
sorted_map = sort(map_contributions)
n_map = length(sorted_map)
map_gini = sum((2 .* (1:n_map) .- n_map .- 1) .* sorted_map) / (n_map * sum(sorted_map))

# Constrained OCS metrics
const_selected = solutions_df.constrained_contribution .> threshold
const_contributions = solutions_df.constrained_contribution[const_selected] .* solutions_df.standardized_bv[const_selected]

const_mean = mean(const_contributions)
const_max = maximum(const_contributions)
const_conc = sum(sort(const_contributions, rev=true)[1:min(Int(ceil(length(const_contributions)*0.1)), length(const_contributions))]) / sum(const_contributions) * 100

# Gini coefficient for Constrained
sorted_const = sort(const_contributions)
n_const = length(sorted_const)
const_gini = sum((2 .* (1:n_const) .- n_const .- 1) .* sorted_const) / (n_const * sum(sorted_const))

println("✓ Portfolio metrics calculated")

# ============================================================================
# CREATE COMPREHENSIVE FIGURE
# ============================================================================

println("\nCreating comprehensive dual metrics figure...")

# Sort robustness data by breeding value
sort!(robustness_df, :breeding_value, rev=true)

# Quartiles for robustness
q25 = quantile(robustness_df.robustness_score, 0.25)
q50 = quantile(robustness_df.robustness_score, 0.50)
q75 = quantile(robustness_df.robustness_score, 0.75)

# Create 1x3 layout (removed redundant panel b, now have a, b, c)
# Create 2x2 layout (4 panels: a, b, c, d)
p = plot(layout=(2, 2), size=(1400, 1000), dpi=300, margin=10Plots.mm, bottom_margin=15Plots.mm)

# ============================================================================
# PANEL A: Individual Robustness Scores (Bar Plot)
# ============================================================================

selected_mask = robustness_df.selected_in_map
unselected_mask = .!robustness_df.selected_in_map
x_pos = 1:nrow(robustness_df)

# Unselected (gray)
if sum(unselected_mask) > 0
    bar!(p[1], x_pos[unselected_mask], robustness_df.robustness_score[unselected_mask],
         color=:lightgray, label="Not Selected", alpha=0.9, linewidth=0, linecolor=:match)
end

# Selected (dark blue - high contrast)
if sum(selected_mask) > 0
    bar!(p[1], x_pos[selected_mask], robustness_df.robustness_score[selected_mask],
         color=:darkblue, label="Selected by MAP-OCS", alpha=1.0, linewidth=0, linecolor=:match)
end

# Reference lines
hline!(p[1], [mean(robustness_df.robustness_score)], 
       color=:red, linestyle=:dash, linewidth=2, label="Mean")
hline!(p[1], [q25, q75], color=:orange, linestyle=:dot, linewidth=1.5, label="Q1/Q3")

xlabel!(p[1], "Individuals (Ranked by Breeding Value)")
ylabel!(p[1], "Robustness Score")
title!(p[1], "a", titlefontsize=14, titlelocation=:left, titlefont=font(14, "Computer Modern"))
plot!(p[1], legend=:topright, legendfontsize=8)

# ============================================================================
# PANEL B: Robustness by Quartile (Scatter Plot) - NEW!
# ============================================================================

# Assign quartiles
robustness_df.quartile = map(robustness_df.robustness_score) do score
    if score < q25
        "Q1"
    elseif score < q50
        "Q2"
    elseif score < q75
        "Q3"
    else
        "Q4"
    end
end

quartile_colors = Dict(
    "Q1" => :indianred,
    "Q2" => :orange,
    "Q3" => :lightblue,
    "Q4" => :steelblue
)

# Plot each quartile
for q in ["Q1", "Q2", "Q3", "Q4"]
    mask = robustness_df.quartile .== q
    if sum(mask) > 0
        for (i, idx) in enumerate(findall(mask))
            marker_shape = robustness_df.selected_in_map[idx] ? :circle : :xcross
            scatter!(p[2], [idx], [robustness_df.robustness_score[idx]],
                    color=quartile_colors[q],
                    markershape=marker_shape,
                    markersize=2.5,
                    alpha=0.8,
                    label=(i == 1 ? "$q" : ""),
                    linewidth=0)
        end
    end
end

# Mean line
hline!(p[2], [mean(robustness_df.robustness_score)], 
       color=:black, linestyle=:dash, linewidth=2, label="")

xlabel!(p[2], "Individuals (Ranked by Breeding Value)")
ylabel!(p[2], "Robustness Score")
title!(p[2], "b", titlefontsize=14, titlelocation=:left, titlefont=font(14, "Computer Modern"))
plot!(p[3], legend=:topright, legendfontsize=7)

# ============================================================================
# PANEL C: Contribution Distribution Metrics (Bar Plot)
# ============================================================================

# Only use 3 metrics (remove Concentration which has different scale)
metric_names = ["Mean\nContribution", "Max\nContribution", "Gini\nCoefficient"]
map_values = [map_mean, map_max, map_gini]
const_values = [const_mean, const_max, const_gini]

# Calculate percent changes
pct_changes = ((const_values .- map_values) ./ map_values) .* 100

x_metrics = 1:length(metric_names)

# Create grouped bar plot (now in position [2])
bar!(p[3], x_metrics .- 0.2, map_values,
     bar_width=0.35,
     color=:lightcoral,
     label="MAP-OCS",
     alpha=0.8)

bar!(p[3], x_metrics .+ 0.2, const_values,
     bar_width=0.35,
     color=:steelblue,
     label="Constrained OCS",
     alpha=0.8)

# Add percentage change annotations
for (i, pct) in enumerate(pct_changes)
    y_pos = max(map_values[i], const_values[i]) * 1.1
    annotate!(p[4], i, y_pos, 
             text(@sprintf("%.1f%%", pct), :center, 9, 
                  pct < 0 ? :green : :red))
end

plot!(p[3], xticks=(x_metrics, metric_names),
      xrotation=0,
      xlabel="Portfolio Vulnerability Metric",
      ylabel="Metric Value",
      title="d", titlefontsize=14, titlelocation=:left, titlefont=font(14, "Computer Modern"),
      legend=:bottomleft,  # Changed from :topright to :bottomleft
      legendfontsize=8)

# ============================================================================
# PANEL C: Contribution Distribution (Violin Plots)
# ============================================================================

# Create data for violin plots
map_data = DataFrame(
    contribution = map_contributions,
    solution = fill("MAP-OCS", length(map_contributions))
)

const_data = DataFrame(
    contribution = const_contributions,
    solution = fill("Constrained", length(const_contributions))
)

combined_data = vcat(map_data, const_data)

# Violin plot (now in position [3])
@df combined_data violin!(p[4], :solution, :contribution,
                          fillcolor=[:lightcoral :steelblue],
                          fillalpha=0.6,
                          linewidth=2,
                          label="")

# Add box plots on top
@df combined_data boxplot!(p[4], :solution, :contribution,
                           fillcolor=[:white :white],
                           fillalpha=0.8,
                           linewidth=2,
                           label="",
                           whisker_width=0.5)

xlabel!(p[4], "")
ylabel!(p[4], "Individual Contribution (c × BV)")
title!(p[4], "c", titlefontsize=14, titlelocation=:left, titlefont=font(14, "Computer Modern"))

# Add statistics annotations to LEFT of each violin
# Violin plot orders alphabetically: "Constrained" (pos 1), "MAP-OCS" (pos 2)
# So LEFT violin = Constrained, RIGHT violin = MAP-OCS

const_stats = @sprintf("Constrained\nn=%d\nμ=%.4f\nσ=%.4f", 
                       length(const_contributions), 
                       mean(const_contributions), 
                       std(const_contributions))
map_stats = @sprintf("MAP-OCS\nn=%d\nμ=%.4f\nσ=%.4f", 
                     length(map_contributions), 
                     mean(map_contributions), 
                     std(map_contributions))

# Place text to the LEFT of each violin at mid-height
y_mid = maximum(combined_data.contribution) * 0.5

# Constrained text to LEFT of position 1 (left violin = Constrained)
annotate!(p[4], 0.65, y_mid,
         text(const_stats, :left, 9))  
# MAP-OCS text to LEFT of position 2 (right violin = MAP-OCS)
annotate!(p[4], 1.65, y_mid,
         text(map_stats, :left, 9))

# ============================================================================
# SAVE FIGURE
# ============================================================================

# Save
output_file = joinpath(OUTPUT_DIR, "dual_metrics_comprehensive_figure.pdf")
savefig(p, output_file)
println("✓ Saved: $output_file")

output_file_png = joinpath(OUTPUT_DIR, "dual_metrics_comprehensive_figure.png")
savefig(p, output_file_png)
println("✓ Saved: $output_file_png")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

println("\n" * "="^80)
println("DUAL METRICS SUMMARY")
println("="^80)

println("\nMETRIC 1: Individual Importance (MCMC-Based)")
selected_rob = robustness_df.robustness_score[robustness_df.selected_in_map]
unselected_rob = robustness_df.robustness_score[.!robustness_df.selected_in_map]

println("  Selected individuals (n=$(sum(robustness_df.selected_in_map))):")
println("    Mean robustness: $(round(mean(selected_rob), digits=6))")
println("    Std dev: $(round(std(selected_rob), digits=6))")

if length(unselected_rob) > 0
    println("  Unselected individuals (n=$(sum(.!robustness_df.selected_in_map))):")
    println("    Mean robustness: $(round(mean(unselected_rob), digits=6))")
    println("    Std dev: $(round(std(unselected_rob), digits=6))")
end

println("\nMETRIC 2: Portfolio Vulnerability")
println("  Mean Contribution:")
println("    MAP-OCS: $(round(map_mean, digits=6))")
println("    Constrained: $(round(const_mean, digits=6))")
println("    Change: $(round(pct_changes[1], digits=2))%")

println("  Maximum Contribution:")
println("    MAP-OCS: $(round(map_max, digits=6)) ($(round(map_max/sum(map_contributions)*100, digits=2))% of total)")
println("    Constrained: $(round(const_max, digits=6)) ($(round(const_max/sum(const_contributions)*100, digits=2))% of total)")
println("    Change: $(round(pct_changes[2], digits=2))%")

println("  Gini Coefficient:")
println("    MAP-OCS: $(round(map_gini, digits=4))")
println("    Constrained: $(round(const_gini, digits=4))")
println("    Change: $(round(pct_changes[3], digits=2))%")

println("  Portfolio Concentration (Top 10%) [not shown in figure]:")
println("    MAP-OCS: $(round(map_conc, digits=2))%")
println("    Constrained: $(round(const_conc, digits=2))%")

println("\n" * "="^80)
println("VISUALIZATION COMPLETE!")
println("="^80)
println("\nComprehensive dual metrics figure created!")
println("  Output: $output_file")
println("="^80)
