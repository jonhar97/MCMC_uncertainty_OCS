"""
Composite Figure — CVaR-OCS Analysis: QTLMAS 2010 Simulation
=============================================================

Produces a single 4-panel publication figure from pre-computed CSVs:

  (a) Risk-return frontier: E[gain] vs CVaR₉₅ across α and μ
  (b) EBV vs contribution — MAP-OCS vs CVaR-OCS vs Oracle (three-way)
  (c) TBV vs contribution — MAP-OCS vs CVaR-OCS vs Oracle (three-way)
  (d) Gain distribution across MCMC scenarios (MAP vs CVaR)

REQUIREMENTS:
  Run cvar_ocs_qtlmas2010_5.jl
      → Save6/cvar_ocs_frontier.csv
      → Save6/cvar_ocs_solutions.csv
  Run QTLMAS_mcmc_robustness_2.jl
      → robustness_analysis_qtlmas.csv
      → evaluation_gains_qtlmas.csv

USAGE:
  include("figure_cvar_qtlmas_composite.jl")
  fig = build_composite_figure_qtlmas()

Author: Jon Ahlinder (Skogforsk) / Ahlinder & Waldmann
"""

using CSV, DataFrames, Statistics, Plots, StatsPlots, LaTeXStrings, Printf, Measures

# =============================================================================
# CONFIGURATION
# =============================================================================

SPECIES_LABEL  = "QTLMAS 2010 Simulation"
THETA          = 0.03
CVAR_LABEL_USE = "CVaR_a0.95_mu_best"   # label for the best CVaR point used

BASE_DIR  = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\QTLMAS\\"
# NOTE: cvar_ocs_qtlmas2010_5.jl writes to Save6
# QTLMAS_mcmc_robustness_2.jl reads from Save5 (check which is current)
# Update SAVE_DIR here to whichever folder contains your most recent CSVs
SAVE_DIR  = joinpath(BASE_DIR, "Save6\\Theta3")
FIG_DIR   = joinpath(BASE_DIR, "Figures")
mkpath(FIG_DIR)

# Input files
FRONTIER_FILE  = joinpath(SAVE_DIR, "cvar_ocs_frontier.csv")
SOLUTIONS_FILE = joinpath(SAVE_DIR, "cvar_ocs_solutions.csv")
ROBUSTNESS_FILE= joinpath(SAVE_DIR, "robustness_analysis_qtlmas.csv")
EVAL_FILE      = joinpath(SAVE_DIR, "evaluation_gains_qtlmas.csv")

# Output
OUTPUT_PDF = joinpath(FIG_DIR, "Figure_CVaR_QTLMAS.pdf")
OUTPUT_PNG = joinpath(FIG_DIR, "Figure_CVaR_QTLMAS.png")

# Selection threshold for contributions
const SELECTION_THRESHOLD = 1e-4

# Colour palette — consistent with spruce figure
const C_MAP    = RGB(0.267, 0.447, 0.690)   # muted blue
const C_CVAR   = RGB(0.172, 0.627, 0.172)   # muted green
const C_ORACLE = RGB(0.576, 0.337, 0.659)   # muted purple

const C_SHARED   = RGB(0.267, 0.447, 0.690)
const C_MAPONLY  = RGB(0.769, 0.306, 0.322)
const C_CVARONLY = RGB(0.172, 0.627, 0.172)

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

# =============================================================================
# HELPERS
# =============================================================================

"""
Find the best CVaR solution for a given alpha (highest CVaR95 in frontier).
Returns the label string.
"""
function best_cvar_label(frontier::DataFrame, alpha::Float64)
    sub = filter(r -> !ismissing(r.alpha) && r.alpha ≈ alpha, frontier)
    isempty(sub) && error("No frontier points for alpha=$alpha")
    idx = argmax(sub.cvar95_eval)
    return sub.label[idx]
end

"""
Extract contribution vector for a given solution label from solutions DataFrame.
Returns Dict(individual_id::Int => contribution::Float64).
"""
function extract_contributions(solutions::DataFrame, label::String)
    row = filter(r -> r.label == label, solutions)
    isempty(row) && error("Label '$label' not found in solutions")
    id_cols = [c for c in names(solutions) if startswith(string(c), "ID_")]
    return Dict(parse(Int, replace(string(c), "ID_" => "")) =>
                Float64(row[1, c]) for c in id_cols)
end

# =============================================================================
# PANEL (a): Risk-return frontier
# =============================================================================
function panel_frontier_qtlmas(frontier::DataFrame)
    map_row = filter(r -> r.label == "MAP-OCS", frontier)[1, :]
    map_cvar = map_row.cvar95_eval

    pa = plot(
        xlabel        = "Genetic gain loss (%)",
        ylabel        = L"\mathrm{CVaR}_{95}\ \mathrm{improvement\ (\%)}",
        title         = "",
        legend        = :topright,
        grid          = false,
        framestyle    = :box,
        titlefontsize = 13,
        guidefontsize = 13,
        tickfontsize  = 12,
        legendfontsize= 10,
        left_margin   = 6Plots.mm,
        bottom_margin = 5Plots.mm,
    )

    for α in ALPHA_LEVELS
        sub = sort(filter(r -> !ismissing(r.alpha) && r.alpha ≈ α, frontier), :mu)
        isempty(sub) && continue
        col = ALPHA_COLORS[α]

        gain_loss_pct = (map_row.gain_exp .- sub.gain_exp) ./ map_row.gain_exp .* 100
        cvar_imp_pct  = (sub.cvar95_eval .- map_cvar) ./ map_cvar .* 100

        plot!(pa, gain_loss_pct, cvar_imp_pct,
              color=col, linewidth=2, label=ALPHA_LABELS[α],
              marker=:circle, markersize=4,
              markerstrokewidth=0, markeralpha=0.7)

        # Elbow: max perpendicular distance from chord (first mu point -> last mu point)
        x0, y0 = gain_loss_pct[1], cvar_imp_pct[1]
        x1, y1 = gain_loss_pct[end], cvar_imp_pct[end]
        chord_len = sqrt((x1 - x0)^2 + (y1 - y0)^2)
        dists = [abs((y1 - y0)*(gx - x0) - (x1 - x0)*(gy - y0)) / chord_len
                 for (gx, gy) in zip(gain_loss_pct, cvar_imp_pct)]
        elbow_idx = argmax(dists)

        scatter!(pa, [gain_loss_pct[elbow_idx]], [cvar_imp_pct[elbow_idx]],
                 marker=:diamond, markersize=9, color=:white,
                 markerstrokewidth=2, markerstrokecolor=col, label="")
    end

    annotate!(pa, :topleft, text("a", 18, :black))

    return pa
end

# =============================================================================
# PANEL (b): EBV vs contribution — MAP vs CVaR vs Oracle (three-way)
# =============================================================================

function panel_ebv_contribution(rob_df::DataFrame, frontier::DataFrame,
                                 solutions::DataFrame)
    # Use best α=0.95 CVaR solution
    best_label = best_cvar_label(frontier, 0.95)

    pb = plot(
        xlabel        = "Standardised EBV index",
        ylabel        = "Contribution (c)",
        title         = "",
        legend        = :left,
        grid          = false,
        framestyle    = :box,
        titlefontsize = 13,
        guidefontsize = 13,
        tickfontsize  = 12,
        legendfontsize= 10,
        left_margin   = 6Plots.mm,
        bottom_margin = 5Plots.mm,
    )

    # Plot each group from rob_df
    grp_specs = [
        ("Shared (MAP+CVaR)",     C_SHARED,   :circle,  5,  0.55, 0,   "Shared"),
        ("MAP-only (dropped)",    C_MAPONLY,  :xcross,  9,  0.90, 0.5, "MAP-only (dropped)"),
        ("CVaR-only (recruited)", C_CVARONLY, :diamond, 9,  0.90, 0.5, "CVaR-only (recruited)"),
    ]

    # MAP contributions from rob_df
    for (grp, col, mk, ms, al, sw, lbl) in grp_specs
        sub = filter(r -> r.selection_group == grp, rob_df)
        isempty(sub) && continue
        scatter!(pb, sub.ebv_index, sub.map_contribution,
                 color=col, alpha=al, markersize=ms, marker=mk,
                 markerstrokewidth=sw, markerstrokecolor=:white,
                 label=lbl)
    end

    # Oracle individuals — if column exists in rob_df
    if hasproperty(rob_df, :selected_oracle)
        oracle_sub = filter(r -> r.selected_oracle, rob_df)
        if !isempty(oracle_sub)
            scatter!(pb, oracle_sub.ebv_index, oracle_sub.oracle_contribution,
                     color=C_ORACLE, alpha=0.7, markersize=7, marker=:utriangle,
                     markerstrokewidth=0, label="Oracle (MAP-TBV)")
        end
    end

    annotate!(pb, :topleft, text("b", 18, :black))
    return pb
end

# =============================================================================
# PANEL (c): TBV vs contribution — three-way MAP / CVaR / Oracle
# =============================================================================

function panel_tbv_contribution(rob_df::DataFrame)
    pc = plot(
        xlabel        = "Standardised TBV index",
        ylabel        = "Contribution (c)",
        title         = "",
        legend        = :left,
        grid          = false,
        framestyle    = :box,
        titlefontsize = 13,
        guidefontsize = 13,
        tickfontsize  = 12,
        legendfontsize= 10,
        left_margin   = 6Plots.mm,
        bottom_margin = 5Plots.mm,
    )

    # Check what columns are available
    has_tbv   = hasproperty(rob_df, :tbv_index)
    has_cvar  = hasproperty(rob_df, :cvar_contribution)
    has_orc   = hasproperty(rob_df, :oracle_contribution) &&
                hasproperty(rob_df, :selected_oracle)

    if !has_tbv
        @warn "tbv_index column not found in robustness CSV — panel (c) will be empty"
        annotate!(pc, :topleft, text("c", 18, :black))
        return pc
    end

    grp_specs = [
        ("Shared (MAP+CVaR)",     C_SHARED,   :circle,  5, 0.55, 0,   "MAP-OCS (shared)"),
        ("MAP-only (dropped)",    C_MAPONLY,  :xcross,  9, 0.90, 0.5, "MAP-only (dropped)"),
        ("CVaR-only (recruited)", C_CVARONLY, :diamond, 9, 0.90, 0.5, "CVaR-only (recruited)"),
    ]

    for (grp, col, mk, ms, al, sw, lbl) in grp_specs
        sub = filter(r -> r.selection_group == grp, rob_df)
        isempty(sub) && continue
        y_col = has_cvar ? sub.cvar_contribution : sub.map_contribution
        scatter!(pc, sub.tbv_index, y_col,
                 color=col, alpha=al, markersize=ms, marker=mk,
                 markerstrokewidth=sw, markerstrokecolor=:white,
                 label=lbl)
    end

    if has_orc
        oracle_sub = filter(r -> r.selected_oracle, rob_df)
        if !isempty(oracle_sub)
            scatter!(pc, oracle_sub.tbv_index, oracle_sub.oracle_contribution,
                     color=C_ORACLE, alpha=0.75, markersize=7, marker=:utriangle,
                     markerstrokewidth=0, label="Oracle (MAP-TBV)")
        end
    end

    annotate!(pc, :topleft, text("c", 18, :black))
    return pc
end

# =============================================================================
# PANEL (d): Gain distribution across MCMC scenarios
# =============================================================================

function panel_gain_distribution_qtlmas(eval_df::DataFrame)
    g_map  = eval_df[eval_df.solution .== "MAP-OCS",  :gain]
    g_cvar = eval_df[eval_df.solution .== "CVaR-OCS", :gain]

    isempty(g_map)  && error("No MAP-OCS rows in evaluation data")
    isempty(g_cvar) && error("No CVaR-OCS rows in evaluation data")

    cvar95_m = mean(g_map[g_map   .< quantile(g_map,  0.05)])
    cvar95_c = mean(g_cvar[g_cvar .< quantile(g_cvar, 0.05)])
    var95_m  = quantile(g_map,  0.05)
    var95_c  = quantile(g_cvar, 0.05)

    pd = plot(
        xlabel        = "Genetic gain (in-sample)",
        ylabel        = "Density",
        title         = "",
        legend        = :bottomright,
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

    annotate!(pd, :topleft, text("d", 18, :black))
    return pd
end

# =============================================================================
# MAIN: ASSEMBLE COMPOSITE FIGURE
# =============================================================================

function build_composite_figure_qtlmas()
    println("=" ^ 70)
    println("COMPOSITE FIGURE — CVaR-OCS QTLMAS Simulation")
    println("=" ^ 70)

    println("\n[1] Loading frontier...")
    frontier = CSV.read(FRONTIER_FILE, DataFrame)
    println("    $(nrow(frontier)) frontier points")

    println("[2] Loading solutions...")
    solutions = CSV.read(SOLUTIONS_FILE, DataFrame)
    println("    $(nrow(solutions)) solution rows")

    println("[3] Loading robustness scores...")
    rob_df = CSV.read(ROBUSTNESS_FILE, DataFrame)
    println("    $(nrow(rob_df)) individuals")

    println("[4] Loading gain evaluation data...")
    eval_df = CSV.read(EVAL_FILE, DataFrame)
    println("    $(nrow(eval_df)) rows")

    # Report available columns for debugging
    println("\n    Robustness CSV columns: ", join(names(rob_df), ", "))

    # Build panels
    println("\n[5] Building panels...")
    pa = panel_frontier_qtlmas(frontier)
    pb = panel_ebv_contribution(rob_df, frontier, solutions)
    pc = panel_tbv_contribution(rob_df)
    pd = panel_gain_distribution_qtlmas(eval_df)

    # Assemble 2×2 layout
    fig = plot(pa, pb, pc, pd,
               layout        = (2, 2),
               size          = (1200, 900),
               dpi           = 300,
               plot_title    = "",
               left_margin   = 8Plots.mm,
               bottom_margin = 8Plots.mm,
               top_margin    = 6Plots.mm,
               right_margin  = 4Plots.mm)

    println("\n[6] Saving figure...")
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
COMPOSITE FIGURE SCRIPT — CVaR-OCS QTLMAS Simulation
======================================================================
Panels:
  (a) Risk-return frontier: E[gain] vs CVaR95 across alpha and mu
  (b) EBV vs contribution — MAP / CVaR / Oracle
  (c) TBV vs contribution — MAP / CVaR / Oracle
  (d) Gain distribution across MCMC scenarios

Requires pre-computed CSVs from:
  - cvar_ocs_qtlmas2010_5.jl
        -> Save6/cvar_ocs_frontier.csv
        -> Save6/cvar_ocs_solutions.csv
  - QTLMAS_mcmc_robustness_2.jl
        -> Save6/robustness_analysis_qtlmas.csv
        -> Save6/evaluation_gains_qtlmas.csv

NOTE: SAVE_DIR mismatch — cvar script uses Save6, robustness script
  defaults to Save5. Ensure both point to the same folder before running.

Run with:
    fig = build_composite_figure_qtlmas()

Output:
  Figures/Figure_CVaR_QTLMAS.pdf
  Figures/Figure_CVaR_QTLMAS.png

NOTE: Panels (b) and (c) depend on robustness CSV having columns:
  selection_group, ebv_index, tbv_index, map_contribution,
  cvar_contribution, oracle_contribution, selected_oracle
  The script prints available columns on load for debugging.
======================================================================
""")
