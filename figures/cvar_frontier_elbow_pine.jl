"""
CVaR-OCS Frontier: Elbow Detection and Publication Figure
==========================================================

Reads the frontier CSV produced by cvar_ocs_forest_trees.jl and generates:

  1. Elbow point per α level (geometric perpendicular-distance method)
  2. Publication-ready 2-panel figure:
       Panel (a) — CVaR₉₅ improvement vs genetic gain loss frontier,
                   one curve per α, elbow points marked
       Panel (b) — Marginal efficiency (ΔCVAR₉₅ / Δgain_loss) vs μ,
                   showing diminishing returns past the elbow

Usage (after running cvar_ocs_forest_trees.jl):
    include("cvar_frontier_elbow.jl")
    results = run_elbow_analysis()

Author: Jon Ahlinder (Skogforsk) / Ahlinder & Waldmann
"""

using CSV, DataFrames, Statistics, LinearAlgebra, Plots, Printf, LaTeXStrings

# =============================================================================
# CONFIGURATION — update paths to match your species
# =============================================================================

SPECIES_LABEL = "Loblolly pine (Pinus taeda)"
THETA         = 0.03

BASE_DIR      = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData\\"
SAVE_DIR      = joinpath(BASE_DIR, "Save", "CVaR_OCS_pine_theta0p03")
FIG_DIR       = joinpath(BASE_DIR, "Figures")

FRONTIER_FILE = joinpath(SAVE_DIR, "cvar_ocs_frontier_pine.csv")
OUTPUT_PDF    = joinpath(FIG_DIR,  "cvar_frontier_elbow_pine.pdf")
OUTPUT_PNG    = joinpath(FIG_DIR,  "cvar_frontier_elbow_pine.png")

# α levels to show (must match what cvar_ocs_forest_trees.jl produced)
ALPHA_LEVELS  = [0.90, 0.95, 0.99]

# Muted colour palette — one per α level
ALPHA_COLORS  = Dict(
    0.90 => RGB(0.267, 0.447, 0.690),   # muted blue
    0.95 => RGB(0.172, 0.627, 0.172),   # muted green
    0.99 => RGB(0.769, 0.306, 0.322),   # muted red
)

# =============================================================================
# ELBOW DETECTION
# =============================================================================

"""
Geometric elbow detection via maximum perpendicular distance.

Given vectors x (gain loss %) and y (CVaR improvement %), both already in
the same units, normalises each to [0,1] and finds the point on the curve
with the greatest perpendicular distance from the line connecting the first
and last points.

Returns: (elbow_index, normalised_distances)
"""
function find_elbow(x::Vector{Float64}, y::Vector{Float64})
    # Normalise to [0, 1]
    x_range = x[end] - x[1]
    y_range = y[end] - y[1]

    x_n = (x .- x[1]) ./ (x_range == 0 ? 1.0 : x_range)
    y_n = (y .- y[1]) ./ (y_range == 0 ? 1.0 : y_range)

    # Direction vector of the chord (first → last)
    dx = x_n[end] - x_n[1]
    dy = y_n[end] - y_n[1]
    chord_len = sqrt(dx^2 + dy^2)

    # Perpendicular distance of each interior point from the chord
    dists = abs.(dy .* (x_n .- x_n[1]) .- dx .* (y_n .- y_n[1])) ./ chord_len

    elbow_idx = argmax(dists)
    return elbow_idx, dists
end

"""
Compute per-α frontier metrics and detect elbow.

Returns a DataFrame with columns:
  alpha, mu, gain_exp, cvar95, gain_loss_pct, cvar_imp_pct, perp_dist, is_elbow
"""
function compute_frontier_metrics(frontier::DataFrame)
    map_row  = filter(r -> r.label == "MAP-OCS", frontier)[1, :]
    map_gain = map_row.gain_exp
    map_cvar = map_row.cvar95_eval

    rows = DataFrame(
        alpha        = Float64[],
        mu           = Float64[],
        label        = String[],
        gain_exp     = Float64[],
        cvar95       = Float64[],
        gain_loss_pct= Float64[],
        cvar_imp_pct = Float64[],
        n_selected   = Int[],
        gini         = Float64[],
        perp_dist    = Float64[],
        is_elbow     = Bool[],
    )

    for α in ALPHA_LEVELS
        sub = sort(filter(r -> !ismissing(r.alpha) && r.alpha == α, frontier), :mu)

        gain_loss = (map_gain .- sub.gain_exp)  ./ map_gain  .* 100
        cvar_imp  = (sub.cvar95_eval .- map_cvar) ./ map_cvar .* 100

        elbow_idx, dists = find_elbow(gain_loss, cvar_imp)

        for (i, row) in enumerate(eachrow(sub))
            push!(rows, (
                α,
                row.mu,
                row.label,
                row.gain_exp,
                row.cvar95_eval,
                gain_loss[i],
                cvar_imp[i],
                row.n_selected,
                row.gini,
                dists[i],
                i == elbow_idx,
            ))
        end
    end

    return rows, map_gain, map_cvar
end

# =============================================================================
# FIGURE
# =============================================================================

"""
Build the 2-panel publication figure.

Panel (a): CVaR_95 improvement (%) vs genetic gain loss (%) — one curve per α,
           y-axis from 0, elbow marked with an open circle on top of the filled
           marker so it stands out clearly.
Panel (b): Marginal efficiency = delta(CVaR_95 imp.) / delta(gain loss) vs mu,
           showing diminishing returns past each elbow. Single shared vertical
           elbow line is drawn at the recommended operating point only to avoid
           crowding; individual elbow mu values annotated on x-axis instead.
"""
function build_figure(metrics::DataFrame)

    alphas = ALPHA_LEVELS

    # LaTeXStrings for axis labels — avoids GKS glyph failures with Unicode
    alpha_labels = Dict(
        0.90 => L"\alpha=0.90",
        0.95 => L"\alpha=0.95",
        0.99 => L"\alpha=0.99",
    )

    # Annotation offsets for elbow labels — tuned per curve to avoid overlap
    # (gain_loss_offset, cvar_imp_offset, halign)
    elbow_offsets = Dict(
        0.90 => ( 0.08, -0.45, :left),
        0.95 => ( 0.08,  0.30, :left),
        0.99 => ( 0.08, -0.45, :left),
    )

    # y-axis range: start at 0, round max up to nearest 2%
    all_cvar_imp = metrics.cvar_imp_pct
    ymax = ceil(maximum(all_cvar_imp) / 2) * 2 + 1.0
    all_gain_loss = metrics.gain_loss_pct
    xmax = ceil(maximum(all_gain_loss) / 2) * 2 + 0.5

    # ── Panel (a): efficiency frontier ────────────────────────────────────────
    pa = plot(
        xlabel        = "Genetic gain loss (%)",
        ylabel        = L"\mathrm{CVaR}_{95}\ \mathrm{improvement\ (\%)}",
        title         = "(a) Gain-tail-risk efficiency frontier",
        legend        = :bottomright,
        grid          = false,
        framestyle    = :box,
        titlefontsize = 11,
        guidefontsize = 10,
        tickfontsize  =  9,
        legendfontsize=  9,
        xlims  = (0.0, xmax),
        ylims  = (0.0, ymax),
        left_margin   = 6Plots.mm,
        bottom_margin = 5Plots.mm,
        right_margin  = 2Plots.mm,
    )

    for α in alphas
        sub = sort(filter(r -> r.alpha == α, metrics), :mu)
        col = ALPHA_COLORS[α]
        lbl = alpha_labels[α]

        # Frontier line + small filled circles at each mu point
        plot!(pa,
            sub.gain_loss_pct, sub.cvar_imp_pct,
            color             = col,
            linewidth         = 2,
            label             = lbl,
            marker            = :circle,
            markersize        = 4,
            markerstrokewidth = 0,
            markeralpha       = 0.6,
        )

        # Elbow: white-filled diamond with colored border — visually distinct
        # from the small filled circles on the line, no compositing artifacts
        elbow = filter(r -> r.is_elbow, sub)[1, :]
        scatter!(pa,
            [elbow.gain_loss_pct], [elbow.cvar_imp_pct],
            color             = :white,
            marker            = :diamond,
            markersize        = 10,
            markerstrokewidth = 2.5,
            markerstrokecolor = col,
            label             = "",
        )

        # Label: "mu=X.XX" offset to avoid overlap with marker
        dx, dy, ha = elbow_offsets[α]
        annotate!(pa,
            elbow.gain_loss_pct + dx,
            elbow.cvar_imp_pct  + dy,
            text(@sprintf("mu=%.2f", elbow.mu), ha, 8, col),
        )
    end

    # ── Panel (b): marginal efficiency ────────────────────────────────────────
    # Compute marginal efficiency for all alpha curves first so we can set ymax
    marg_data = Dict{Float64, Tuple{Vector{Float64}, Vector{Float64}}}()
    elbow_mus = Dict{Float64, Float64}()

    for α in alphas
        sub   = sort(filter(r -> r.alpha == α, metrics), :mu)
        mus   = sub.mu[2:end]
        Δcvar = diff(sub.cvar_imp_pct)
        Δgain = diff(sub.gain_loss_pct)
        marg  = [dg == 0 ? NaN : dc / dg for (dc, dg) in zip(Δcvar, Δgain)]
        # Note: a visible drop in marginal efficiency (e.g. alpha=0.90 at mu=1.50)
        # reflects a genuine change in optimizer solution — n_selected increases
        # and the coancestry constraint becomes binding differently. This is real
        # signal, not noise, and confirms the elbow as the right stopping point.
        marg_data[α] = (mus, marg)

        elbow_row = filter(r -> r.is_elbow, sub)[1, :]
        elbow_mus[α] = elbow_row.mu
    end

    # y-axis for panel (b): clip extreme first-interval values for readability
    all_marg = vcat([collect(skipmissing(v[2])) for v in values(marg_data)]...)
    filter!(isfinite, all_marg)
    marg_ymax = min(ceil(quantile(all_marg, 0.95) * 1.2), maximum(all_marg) * 1.05)
    marg_ymax = ceil(marg_ymax / 0.5) * 0.5   # round to nearest 0.5

    pb = plot(
        xlabel        = L"\mathrm{CVaR\ weight}\ (\mu)",
        ylabel        = L"\Delta\mathrm{CVaR}_{95}\%\ /\ \Delta\mathrm{gain\ loss}\%",
        title         = "(b) Marginal efficiency of CVaR weighting",
        legend        = :topright,
        grid          = false,
        framestyle    = :box,
        titlefontsize = 11,
        guidefontsize = 10,
        tickfontsize  =  9,
        legendfontsize=  9,
        ylims         = (0.0, marg_ymax),
        left_margin   = 8Plots.mm,   # wider for long y-label
        bottom_margin = 5Plots.mm,
        right_margin  = 2Plots.mm,
    )

    for α in alphas
        col  = ALPHA_COLORS[α]
        lbl  = alpha_labels[α]
        mus, marg = marg_data[α]

        plot!(pb,
            mus, marg,
            color             = col,
            linewidth         = 2,
            label             = lbl,
            marker            = :circle,
            markersize        = 5,
            markerstrokewidth = 0,
        )

        # Annotate elbow mu on x-axis with a tick-mark style line rather than
        # crowded vlines — draw a short vertical segment at y=0..0.15*ymax
        eμ = elbow_mus[α]
        plot!(pb,
            [eμ, eμ], [0.0, marg_ymax * 0.12],
            color     = col,
            linestyle = :dash,
            linewidth = 1.2,
            label     = "",
            alpha     = 0.8,
        )
        annotate!(pb,
            eμ, -marg_ymax * 0.06,
            text(@sprintf("%.2f", eμ), :center, 7, col),
        )
    end

    # Combine into 1×2 layout
    fig = plot(pa, pb,
        layout          = (1, 2),
        size            = (980, 440),
        plot_title      = "$SPECIES_LABEL  |  Theta=$(THETA)",
        plot_titlefontsize = 12,
        top_margin      = 8Plots.mm,
    )

    return fig
end

# =============================================================================
# REPORTING
# =============================================================================

function print_elbow_report(metrics::DataFrame, map_gain::Float64, map_cvar::Float64)
    println("\n", "="^70)
    println("ELBOW DETECTION RESULTS — CVaR-OCS Frontier")
    println("="^70)
    println(@sprintf("  MAP-OCS baseline:  gain = %.4f  CVaR₉₅ = %.4f", map_gain, map_cvar))
    println()

    for α in ALPHA_LEVELS
        sub  = sort(filter(r -> r.alpha == α, metrics), :mu)
        elbow = filter(r -> r.is_elbow, sub)[1, :]

        println(@sprintf("  α = %.2f  →  elbow at μ = %.2f", α, elbow.mu))
        println(@sprintf("    Gain loss    : %+.2f%%  (gain = %.4f)",
                         elbow.gain_loss_pct, elbow.gain_exp))
        println(@sprintf("    CVaR₉₅ gain  : %+.2f%%  (CVaR₉₅ = %.4f)",
                         elbow.cvar_imp_pct, elbow.cvar95))
        println(@sprintf("    n selected   : %d   Gini = %.3f",
                         elbow.n_selected, elbow.gini))
        println(@sprintf("    Perp. dist   : %.4f  (higher = sharper elbow)",
                         elbow.perp_dist))
        println()
    end

    # Recommended operating point = α with highest CVaR improvement at elbow,
    # subject to gain loss < 2%
    candidates = filter(r -> r.is_elbow && r.gain_loss_pct < 2.0, metrics)
    if nrow(candidates) > 0
        best = candidates[argmax(candidates.cvar_imp_pct), :]
        println("  ✓ RECOMMENDED OPERATING POINT:")
        println(@sprintf("    α = %.2f, μ = %.2f", best.alpha, best.mu))
        println(@sprintf("    Gain loss = %.2f%%,  CVaR₉₅ improvement = %.2f%%",
                         best.gain_loss_pct, best.cvar_imp_pct))
    end

    println("="^70)
end

# =============================================================================
# MAIN
# =============================================================================

function run_elbow_analysis()
    println("="^70)
    println("CVaR-OCS FRONTIER: ELBOW DETECTION")
    println("$SPECIES_LABEL  |  Θ=$THETA")
    println("="^70)

    # ── Load frontier ─────────────────────────────────────────────────────────
    println("\n[1] Loading frontier: $FRONTIER_FILE")
    frontier = CSV.read(FRONTIER_FILE, DataFrame)
    println(@sprintf("    %d frontier points loaded", nrow(frontier)))

    # ── Compute metrics and elbows ────────────────────────────────────────────
    println("\n[2] Computing elbow points...")
    metrics, map_gain, map_cvar = compute_frontier_metrics(frontier)

    print_elbow_report(metrics, map_gain, map_cvar)

    # ── Build figure ──────────────────────────────────────────────────────────
    println("\n[3] Building figure...")
    fig = build_figure(metrics)

    mkpath(FIG_DIR)
    savefig(fig, OUTPUT_PDF)
    savefig(fig, OUTPUT_PNG)
    println("    ✓ $(OUTPUT_PDF)")
    println("    ✓ $(OUTPUT_PNG)")

    # ── Save metrics CSV ──────────────────────────────────────────────────────
    metrics_file = joinpath(SAVE_DIR, "frontier_metrics_with_elbow.csv")
    CSV.write(metrics_file, metrics)
    println("    ✓ $metrics_file")

    println("\n", "="^70)
    println("DONE")
    println("="^70)

    return Dict(
        "metrics"   => metrics,
        "map_gain"  => map_gain,
        "map_cvar"  => map_cvar,
        "figure"    => fig,
    )
end

# =============================================================================
# PRINT USAGE
# =============================================================================

println("""
======================================================================
CVaR-OCS FRONTIER ELBOW ANALYSIS — READY
======================================================================
Species : $SPECIES_LABEL
Theta   : $THETA
α levels: $(join(ALPHA_LEVELS, ", "))

Run with:
    results = run_elbow_analysis()

Outputs:
  • cvar_frontier_elbow_pine.pdf/.png  — 2-panel figure
  • frontier_metrics_with_elbow.csv      — metrics + elbow flags
======================================================================
""")
