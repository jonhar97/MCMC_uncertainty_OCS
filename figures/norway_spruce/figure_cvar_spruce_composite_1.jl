"""
Composite Figure — CVaR-OCS Analysis: Norway Spruce (Picea abies)
==================================================================

Produces a single 4-panel publication figure from pre-computed CSVs:

  (a) Gain–tail-risk efficiency frontier with elbow markers
  (b) Marginal efficiency of CVaR weighting vs mu
  (c) EBV vs robustness score, coloured by selection group
  (d) Gain distribution across MCMC scenarios (MAP vs CVaR)

REQUIREMENTS:
  Run cvar_ocs_forest_trees.jl  → cvar_ocs_frontier_spruce.csv
  Run cvar_frontier_elbow.jl    → frontier_metrics_with_elbow.csv
  Run forest_robustness_analysis_2.jl → robustness_analysis_spruce.csv
                                      → evaluation_gains_spruce.csv

USAGE:
  include("figure_cvar_spruce_composite.jl")
  fig = build_composite_figure()

Author: Jon Ahlinder (Skogforsk) / Ahlinder & Waldmann
"""

using CSV, DataFrames, Statistics, Plots, StatsPlots, LaTeXStrings, Printf, Measures

# =============================================================================
# CONFIGURATION
# =============================================================================

SPECIES_LABEL = "Norway Spruce (Picea abies)"
THETA         = 0.02
CVAR_LABEL    = "CVaR_a0.90_mu1.25"   # update when re-run with mu=1.25

BASE_DIR   = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\"
SAVE_DIR   = joinpath(BASE_DIR, "Save", "CVaR_OCS_spruce_theta0p02")
FIG_DIR    = joinpath(BASE_DIR, "Figures")

# Input files — all pre-computed
FRONTIER_FILE  = joinpath(SAVE_DIR, "frontier_metrics_with_elbow.csv")   # from cvar_frontier_elbow.jl
ROBUSTNESS_FILE= joinpath(SAVE_DIR, "robustness_analysis_spruce.csv")    # from forest_robustness_analysis_2.jl
EVAL_FILE      = joinpath(SAVE_DIR, "evaluation_gains_spruce.csv")       # from forest_robustness_analysis_2.jl

# Output
OUTPUT_PDF = joinpath(FIG_DIR, "Figure_CVaR_spruce.pdf")
OUTPUT_PNG = joinpath(FIG_DIR, "Figure_CVaR_spruce.png")

# Colour palette — consistent across all scripts
const C_MAP   = RGB(0.267, 0.447, 0.690)   # muted blue
const C_CVAR  = RGB(0.172, 0.627, 0.172)   # muted green

const C_SHARED    = RGB(0.267, 0.447, 0.690)   # blue
const C_MAPONLY   = RGB(0.769, 0.306, 0.322)   # muted red
const C_CVARONLY  = RGB(0.172, 0.627, 0.172)   # green

# Alpha levels to show on frontier
const ALPHA_LEVELS = [0.90, 0.95, 0.99]
const ALPHA_COLORS = Dict(
    0.90 => RGB(0.267, 0.447, 0.690),
    0.95 => RGB(0.172, 0.627, 0.172),
    0.99 => RGB(0.769, 0.306, 0.322),
)
const ALPHA_LABELS = Dict(
    0.90 => L"\alpha=0.90",
    0.95 => L"\alpha=0.95",
    0.99 => L"\alpha=0.99",
)

# Annotation offsets for elbow labels [gain_loss_offset, cvar_imp_offset, halign]
# Elbow label offsets: (x_offset, y_absolute, halign)
# y values are absolute positions on the CVaR improvement (%) axis
const ELBOW_OFFSETS = Dict(
    0.90 => ( 0.08, -0.45, :left),   # relative offset (default)
    0.95 => ( 0.08,  0.30, :left),   # relative offset
    0.99 => ( 0.08, -0.45, :left),   # relative offset
)

# Override absolute y positions for specific alpha labels where
# the automatic offset causes overlap with the curve
const ELBOW_Y_ABSOLUTE = Dict(
    0.95 => 10.5,
    0.99 =>  6.5,
)

# =============================================================================
# PANEL (a): Efficiency frontier
# =============================================================================

function panel_frontier(metrics::DataFrame)
    all_y = metrics.cvar_imp_pct
    all_x = metrics.gain_loss_pct
    ymax  = ceil(maximum(all_y) / 2) * 2 + 1.0
    xmax  = ceil(maximum(all_x) / 2) * 2 + 0.5

    pa = plot(
        xlabel        = "Genetic gain loss (%)",
        ylabel        = L"\mathrm{CVaR}_{95}\ \mathrm{improvement\ (\%)}",
        title         = "",
        legend        = :bottomright,
        grid          = false,
        framestyle    = :box,
        titlefontsize = 13,
        guidefontsize = 13,
        tickfontsize  = 12,
        legendfontsize= 11,
        xlims  = (0.0, xmax),
        ylims  = (0.0, ymax),
        left_margin   = 6Plots.mm,
        bottom_margin = 5Plots.mm,
    )

    for α in ALPHA_LEVELS
        sub = sort(filter(r -> r.alpha == α, metrics), :mu)
        col = ALPHA_COLORS[α]

        plot!(pa, sub.gain_loss_pct, sub.cvar_imp_pct,
              color=col, linewidth=2, label=ALPHA_LABELS[α],
              marker=:circle, markersize=4,
              markerstrokewidth=0, markeralpha=0.6)

        elbow = filter(r -> r.is_elbow, sub)[1, :]

        # White-filled diamond elbow marker
        scatter!(pa, [elbow.gain_loss_pct], [elbow.cvar_imp_pct],
                 color=:white, marker=:diamond, markersize=10,
                 markerstrokewidth=2.5, markerstrokecolor=col, label="")

        dx, dy, ha = ELBOW_OFFSETS[α]
        y_pos = haskey(ELBOW_Y_ABSOLUTE, α) ? ELBOW_Y_ABSOLUTE[α] :
                                               elbow.cvar_imp_pct + dy
        annotate!(pa,
                  elbow.gain_loss_pct + dx,
                  y_pos,
                  text(@sprintf("mu=%.2f", elbow.mu), ha, 11, col))
    end
    # Panel label
    annotate!(pa, :topleft, text("a", 18, :black))
    return pa
end

# =============================================================================
# PANEL (b): Marginal efficiency
# =============================================================================

function panel_marginal(metrics::DataFrame)
    # Pre-compute marginal efficiency per alpha
    marg_data  = Dict{Float64, Tuple{Vector{Float64}, Vector{Float64}}}()
    elbow_mus  = Dict{Float64, Float64}()

    for α in ALPHA_LEVELS
        sub   = sort(filter(r -> r.alpha == α, metrics), :mu)
        mus   = sub.mu[2:end]
        Δcvar = diff(sub.cvar_imp_pct)
        Δgain = diff(sub.gain_loss_pct)
        marg  = [dg == 0 ? NaN : dc / dg for (dc, dg) in zip(Δcvar, Δgain)]
        marg_data[α] = (mus, marg)
        elbow_mus[α] = filter(r -> r.is_elbow, sub)[1, :].mu
    end

    all_marg = vcat([collect(skipmissing(v[2])) for v in values(marg_data)]...)
    filter!(isfinite, all_marg)
    marg_ymax = ceil(min(quantile(all_marg, 0.95) * 1.2,
                         maximum(all_marg) * 1.05) / 0.5) * 0.5

    pb = plot(
        xlabel        = L"\mathrm{CVaR\ weight}\ (\mu)",
        ylabel        = L"\Delta\mathrm{CVaR}_{95}\%\ /\ \Delta\mathrm{gain\ loss}\%",
        title         = "",
        legend        = :topright,
        grid          = false,
        framestyle    = :box,
        titlefontsize = 13,
        guidefontsize = 13,
        tickfontsize  = 12,
        legendfontsize= 11,
        xlims         = (0.0, 5.0),
        ylims         = (0.0, marg_ymax),
        left_margin   = 8Plots.mm,
        bottom_margin = 5Plots.mm,
    )

    for α in ALPHA_LEVELS
        col = ALPHA_COLORS[α]
        mus, marg = marg_data[α]

        plot!(pb, mus, marg, color=col, linewidth=2, label=ALPHA_LABELS[α],
              marker=:circle, markersize=5, markerstrokewidth=0)

        eμ = elbow_mus[α]
        if eμ <= 5.0   # only annotate if within zoomed x-range
            plot!(pb, [eμ, eμ], [0.0, marg_ymax * 0.12],
                  color=col, linestyle=:dash, linewidth=1.2, label="", alpha=0.8)
            annotate!(pb, eμ, -marg_ymax * 0.06,
                      text(@sprintf("%.2f", eμ), :center, 10, col))
        end
    end
    # Panel label
    annotate!(pb, :topleft, text("b", 18, :black))
    return pb
end

# =============================================================================
# PANEL (c): EBV vs robustness score by selection group
# =============================================================================

function panel_ebv_robustness(rob_df::DataFrame)
    grp_colors = Dict(
        "Shared (MAP+CVaR)"      => C_SHARED,
        "MAP-only (dropped)"     => C_MAPONLY,
        "CVaR-only (recruited)"  => C_CVARONLY,
    )
    grp_markers = Dict(
        "Shared (MAP+CVaR)"      => :circle,
        "MAP-only (dropped)"     => :xcross,
        "CVaR-only (recruited)"  => :diamond,
    )

    pc = plot(
        xlabel        = "Standardised EBV index",
        ylabel        = "Robustness score",
        title         = "",
        legend        = :topright,
        grid          = false,
        framestyle    = :box,
        titlefontsize = 13,
        guidefontsize = 13,
        tickfontsize  = 12,
        legendfontsize= 11,
        left_margin   = 6Plots.mm,
        bottom_margin = 5Plots.mm,
    )

    # Draw in order: shared first (background), then map-only and cvar-only on top
    for grp in ["Shared (MAP+CVaR)", "MAP-only (dropped)", "CVaR-only (recruited)"]
        sub = filter(r -> r.selection_group == grp, rob_df)
        isempty(sub) && continue
        col = grp_colors[grp]
        mk  = grp_markers[grp]
        ms  = grp == "Shared (MAP+CVaR)" ? 5 : 8
        al  = grp == "Shared (MAP+CVaR)" ? 0.5 : 0.85
        sw  = grp == "Shared (MAP+CVaR)" ? 0 : 0.5

        scatter!(pc, sub.ebv_index, sub.robustness_score,
                 color=col, alpha=al, markersize=ms, marker=mk,
                 markerstrokewidth=sw, markerstrokecolor=:white,
                 label=grp)
    end
    # Panel label — placed in upper-left corner of the axes
    annotate!(pc, :topleft, text("c", 18, :black))
    return pc
end

# =============================================================================
# PANEL (d): Gain distribution across MCMC scenarios
# =============================================================================

function panel_gain_distribution(eval_df::DataFrame)
    g_map  = eval_df[eval_df.solution .== "MAP-OCS",  :gain]
    g_cvar = eval_df[eval_df.solution .== "CVaR-OCS", :gain]

    cvar95_m = mean(g_map[g_map   .< quantile(g_map,  0.05)])
    cvar95_c = mean(g_cvar[g_cvar .< quantile(g_cvar, 0.05)])
    var95_m  = quantile(g_map,  0.05)
    var95_c  = quantile(g_cvar, 0.05)

    pd = plot(
        xlabel        = "Genetic gain (in-sample)",
        ylabel        = "Density",
        title         = "",
        legend        = :topright,
        grid          = false,
        framestyle    = :box,
        titlefontsize = 13,
        guidefontsize = 13,
        tickfontsize  = 12,
        legendfontsize= 9,
        left_margin   = 6Plots.mm,
        bottom_margin = 5Plots.mm,
    )

    density!(pd, g_map,  color=C_MAP,  lw=2.5,
             label=@sprintf("MAP-OCS (CVaR95=%.3f)", cvar95_m))
    density!(pd, g_cvar, color=C_CVAR, lw=2.5,
             label=@sprintf("CVaR-OCS (CVaR95=%.3f)", cvar95_c))
    vline!(pd, [var95_m], color=C_MAP,  lw=1.5, ls=:dash,
           label=@sprintf("VaR95 MAP=%.3f",  var95_m))
    vline!(pd, [var95_c], color=C_CVAR, lw=1.5, ls=:dash,
           label=@sprintf("VaR95 CVaR=%.3f", var95_c))

    # Panel label
    annotate!(pd, :topleft, text("d", 18, :black))
    return pd
end

# =============================================================================
# MAIN: ASSEMBLE COMPOSITE FIGURE
# =============================================================================

function build_composite_figure()
    println("=" ^ 70)
    println("COMPOSITE FIGURE — CVaR-OCS Norway Spruce")
    println("=" ^ 70)

    # Load data
    println("\n[1] Loading frontier metrics...")
    metrics = CSV.read(FRONTIER_FILE, DataFrame)
    # Ensure is_elbow is Bool
    if eltype(metrics.is_elbow) != Bool
        metrics.is_elbow = Bool.(metrics.is_elbow)
    end
    println("    $(nrow(metrics)) frontier points loaded")

    println("[2] Loading robustness scores...")
    rob_df = CSV.read(ROBUSTNESS_FILE, DataFrame)
    println("    $(nrow(rob_df)) individuals loaded")

    println("[3] Loading gain evaluation data...")
    eval_df = CSV.read(EVAL_FILE, DataFrame)
    println("    $(nrow(eval_df)) rows loaded")

    # Build panels
    println("\n[4] Building panels...")
    pa = panel_frontier(metrics)
    pb = panel_marginal(metrics)
    pc = panel_ebv_robustness(rob_df)
    pd = panel_gain_distribution(eval_df)

    # Assemble 2×2 layout
    fig = plot(pa, pb, pc, pd,
               layout        = (2, 2),
               size          = (1200, 900),
               dpi           = 300,
               left_margin   = 8Plots.mm,
               bottom_margin = 8Plots.mm,
               top_margin    = 8Plots.mm,
               right_margin  = 4Plots.mm,
               plot_title    = "",
               plot_titlefontsize = 12,
               titlefontsize      = 13)

    # Save
    println("\n[5] Saving figure...")
    mkpath(FIG_DIR)
    savefig(fig, OUTPUT_PDF)
    savefig(fig, OUTPUT_PNG)
    println("    \u2713 $(OUTPUT_PDF)")
    println("    \u2713 $(OUTPUT_PNG)")

    println("\n" * "=" ^ 70)
    println("DONE")
    println("=" ^ 70)

    return fig
end

# =============================================================================
# PRINT USAGE
# =============================================================================

println("""
======================================================================
COMPOSITE FIGURE SCRIPT — CVaR-OCS Norway Spruce
======================================================================
Panels:
  (a) Gain-tail-risk efficiency frontier with elbow markers
  (b) Marginal efficiency of CVaR weighting
  (c) EBV vs robustness score by selection group
  (d) Gain distribution across MCMC scenarios

Requires pre-computed CSVs from:
  - cvar_frontier_elbow.jl          -> frontier_metrics_with_elbow.csv
  - forest_robustness_analysis_2.jl -> robustness_analysis_spruce.csv
                                    -> evaluation_gains_spruce.csv

Run with:
    fig = build_composite_figure()

Output:
  Figures/Figure_CVaR_spruce.pdf
  Figures/Figure_CVaR_spruce.png
======================================================================
""")
