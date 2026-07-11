# =============================================================================
# Plotting Script — QTL-MAS 2010 Bivariate GBLUP Results
# =============================================================================
#
# Reads outputs from JWASGBLUP_QTLMAS2010.jl and produces publication figures:
#
#   Fig 1a: EBV vs TBV scatter — y1 (continuous), unphenotyped candidates (n=900)
#   Fig 1b: EBV vs TBV scatter — y2 (binary liability), unphenotyped candidates
#   Fig 2:  Posterior SD (sqrt PEV) vs EBV — both traits, unphenotyped only
#   Fig 3:  Heritability posteriors — violin or bar with HPD interval
#   Fig 4:  Genetic correlation posterior — histogram from MCMC samples
#
# Figures follow manuscript style:
#   - Panel labels (a, b) not titles
#   - No background grid
#   - Muted palette: blue (#4878CF), orange (#E8801A), green (#6ACC65)
#   - GP regression line (Matern 3/2) with 95% credible band where applicable
#   - Publication size: 90mm (single col) or 180mm (double col)
# =============================================================================

using CSV, DataFrames, Statistics, LinearAlgebra, Printf, Distributions
using Plots, StatsPlots
using AbstractGPs, KernelFunctions, Optim

gr()   # GR backend for publication-quality output

# =============================================================================
# 0. CONFIGURATION — point to SAVE_DIR from the JWAS script
# =============================================================================

SAVE_DIR = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\QTLMAS\\Save6\\"
FIG_DIR  = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\QTLMAS\\Figures\\"
mkpath(FIG_DIR)

# Colour palette (muted, publication-friendly)
COL_BLUE   = RGB(72/255,  120/255, 207/255)   # #4878CF
COL_ORANGE = RGB(232/255, 128/255,  26/255)   # #E8801A
COL_GREEN  = RGB(106/255, 204/255, 101/255)   # #6ACC65
COL_RED    = RGB(210/255,  77/255,  77/255)
COL_GREY   = RGB(160/255, 160/255, 160/255)

# Figure dimensions (mm → pixels at 300 dpi)
MM_TO_PX  = 300 / 25.4
W_SINGLE  = round(Int, 90  * MM_TO_PX)   # 90 mm single column
W_DOUBLE  = round(Int, 180 * MM_TO_PX)   # 180 mm double column
H_SQUARE  = round(Int, 80  * MM_TO_PX)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
println("Loading results from SAVE_DIR...")

# Read directly from JWAS output .txt files
ebv_y1 = CSV.read(joinpath(SAVE_DIR, "EBV_y1.txt"), DataFrame,
                  delim=',', header=true, missingstring=["NA"])
ebv_y2 = CSV.read(joinpath(SAVE_DIR, "EBV_y2.txt"), DataFrame,
                  delim=',', header=true, missingstring=["NA"])
h2     = CSV.read(joinpath(SAVE_DIR, "heritability.txt"), DataFrame,
                  delim=',', header=true, missingstring=["NA"])

# Load TBV reference and identify unphenotyped candidates
tbv_file   = joinpath(BASE_DIR, "tbv.txt")
tbv_raw    = readdlm(tbv_file, ',', header=false)
all_ids    = Int.(tbv_raw[:, 1])
all_tbv_y1 = Float64.(tbv_raw[:, 5])    # col 5: TBV Q trait
all_tbv_y2 = Float64.(tbv_raw[:, 13])   # col 13: TBV B trait liability
pheno_raw  = readdlm(joinpath(BASE_DIR, "phenotypes.txt"), ',', header=false)
phenotyped_ids = Set(Int.(pheno_raw[:, 1]))

# Add TBV and phenotyped flag to EBV dataframes
for (df, tbv_dict) in [(ebv_y1, Dict(all_ids[i]=>all_tbv_y1[i] for i in eachindex(all_ids))),
                        (ebv_y2, Dict(all_ids[i]=>all_tbv_y2[i] for i in eachindex(all_ids)))]
    df[!, :ID_int]    = parse.(Int, string.(df.ID))
    df[!, :TBV]       = [get(tbv_dict, id, NaN) for id in df.ID_int]
    df[!, :Phenotyped] = [id in phenotyped_ids for id in df.ID_int]
end

# Filter to unphenotyped selection candidates (latest generation, n=900)
cand_y1 = filter(r -> !r.Phenotyped, ebv_y1)
cand_y2 = filter(r -> !r.Phenotyped, ebv_y2)
println("  Unphenotyped candidates: $(nrow(cand_y1))")

# Rename EBV/PEV columns and compute posterior SD
rename!(cand_y1, :EBV => :EBV_post_mean)
rename!(cand_y2, :EBV => :EBV_post_mean)
cand_y1[!, :postSD_y1] = sqrt.(Float64.(cand_y1.PEV))
cand_y2[!, :postSD_y2] = sqrt.(Float64.(cand_y2.PEV))
# EBV_post_SD (sqrt PEV) is already in the EBV means files — no join needed
# Rename for convenience in plotting

# =============================================================================
# 2. GP REGRESSION HELPER (Matern 3/2, MLE hyperparameters via Nelder-Mead)
# =============================================================================

function fit_gp_matern32(x::Vector{Float64}, y::Vector{Float64};
                          n_grid::Int=200)
    x_mean, x_std = mean(x), std(x)
    y_mean, y_std = mean(y), std(y)
    xn = (x .- x_mean) ./ x_std
    yn = (y .- y_mean) ./ y_std

    # MLE for log(σ_f), log(l), log(σ_n) via Nelder-Mead
    function neg_lml(params)
        σ_f, l, σ_n = exp.(params)
        k   = σ_f^2 * with_lengthscale(Matern32Kernel(), l)
        gp  = GP(k)
        fx  = gp(xn, σ_n^2)
        return -logpdf(fx, yn)
    end
    res    = optimize(neg_lml, [0.0, 0.0, -1.0], NelderMead())
    σ_f, l, σ_n = exp.(Optim.minimizer(res))

    k  = σ_f^2 * with_lengthscale(Matern32Kernel(), l)
    gp = GP(k)
    fx = gp(xn, σ_n^2)
    p_fx = posterior(fx, yn)

    x_grid_n = collect(range(minimum(xn) - 0.1, maximum(xn) + 0.1, length=n_grid))
    ms = mean(p_fx(x_grid_n, 1e-6))
    vs = var(p_fx(x_grid_n, 1e-6))

    x_grid = x_grid_n .* x_std .+ x_mean
    m_out  = ms .* y_std .+ y_mean
    sd_out = sqrt.(vs) .* y_std

    # R² on original scale
    y_pred_train = mean(p_fx(xn, 1e-6)) .* y_std .+ y_mean
    r2 = 1 - sum((y .- y_pred_train).^2) / sum((y .- mean(y)).^2)

    return x_grid, m_out, sd_out, r2, σ_f, l, σ_n
end

# =============================================================================
# 3. FIGURE 1: EBV vs TBV scatter (unphenotyped candidates, both traits)
# =============================================================================
println("\nFigure 1: EBV vs TBV scatters...")

function ebv_tbv_panel(ebv_vec, tbv_vec, color, trait_label, panel_label)
    r = cor(ebv_vec, tbv_vec)

    # GP regression
    x_gp, m_gp, sd_gp, r2, = fit_gp_matern32(ebv_vec, tbv_vec)

    p = scatter(ebv_vec, tbv_vec,
        markersize=3, markerstrokewidth=0, alpha=0.45, color=color,
        xlabel="EBV",
        ylabel="TBV",
        legend=false,
        grid=false,
        framestyle=:box,
        tickfontsize=11,
        labelfontsize=12)

    # GP mean ± 1 SD band
    plot!(p, x_gp, m_gp,
        ribbon=sd_gp, fillalpha=0.2,
        color=:black, linewidth=1.5)

    # Annotations: panel label + r
    xl = minimum(ebv_vec) + 0.02 * (maximum(ebv_vec) - minimum(ebv_vec))
    yl = maximum(tbv_vec) - 0.04 * (maximum(tbv_vec) - minimum(tbv_vec))
    annotate!(p, xl, yl,
        text(@sprintf("%s\nr = %.3f", panel_label, r), :left, 11, :black))

    return p
end

p1a = ebv_tbv_panel(cand_y1.EBV_post_mean, cand_y1.TBV, COL_BLUE,  "y1", "a")
p1b = ebv_tbv_panel(cand_y2.EBV_post_mean, cand_y2.TBV, COL_ORANGE, "y2", "b")

fig1 = plot(p1a, p1b,
    layout=(1, 2),
    size=(W_DOUBLE, H_SQUARE),
    left_margin=5Plots.mm, bottom_margin=5Plots.mm)

savefig(fig1, joinpath(FIG_DIR, "Fig1_EBV_vs_TBV_candidates.pdf"))
savefig(fig1, joinpath(FIG_DIR, "Fig1_EBV_vs_TBV_candidates.png"))
println("  Saved Fig1")

# =============================================================================
# 4. FIGURE 2: Posterior SD vs EBV (uncertainty vs breeding value)
# =============================================================================
println("Figure 2: Posterior SD vs EBV...")

function postsd_panel(ebv_vec, sd_vec, color, panel_label)
    p = scatter(ebv_vec, sd_vec,
        markersize=3, markerstrokewidth=0, alpha=0.45, color=color,
        xlabel="Posterior mean EBV",
        ylabel="Posterior SD",
        legend=false,
        grid=false,
        framestyle=:box,
        tickfontsize=11,
        labelfontsize=12)

    annotate!(p,
        minimum(ebv_vec) + 0.02*(maximum(ebv_vec)-minimum(ebv_vec)),
        maximum(sd_vec)  - 0.04*(maximum(sd_vec)-minimum(sd_vec)),
        text(panel_label, :left, 11, :black))
    return p
end

p2a = postsd_panel(cand_y1.EBV_post_mean, cand_y1.postSD_y1, COL_BLUE,   "a")
p2b = postsd_panel(cand_y2.EBV_post_mean, cand_y2.postSD_y2, COL_ORANGE, "b")

fig2 = plot(p2a, p2b,
    layout=(1, 2),
    size=(W_DOUBLE, H_SQUARE),
    left_margin=5Plots.mm, bottom_margin=5Plots.mm)

savefig(fig2, joinpath(FIG_DIR, "Fig2_PostSD_vs_EBV.pdf"))
savefig(fig2, joinpath(FIG_DIR, "Fig2_PostSD_vs_EBV.png"))
println("  Saved Fig2")

# =============================================================================
# 5. FIGURE 3: Heritability estimates (posterior mean + HPD interval)
# =============================================================================
# Reads h2 from variance_components.csv (JWAS heritability table)
# Expected columns: Covariance (trait name), Estimate (posterior mean), SD
println("Figure 3: Heritability estimates...")

# JWAS heritability table typically has columns: Covariance, Estimate, SD
# Rename for clarity if needed
rename_dict = Dict(names(h2) .=> names(h2))   # identity — adjust if columns differ
# JWAS heritability.txt columns: Covariance, Estimate, SD
trait_labels = string.(h2.Covariance)
h2_means     = Float64.(h2.Estimate)
h2_sds       = Float64.(h2.SD)
h2_lo        = max.(0.0, h2_means .- 1.96 .* h2_sds)
h2_hi        = min.(1.0, h2_means .+ 1.96 .* h2_sds)

colors_h2 = [COL_BLUE, COL_ORANGE, COL_GREEN, COL_RED][1:length(trait_labels)]

fig3 = scatter(1:length(trait_labels), h2_means,
    yerror=(h2_means .- h2_lo, h2_hi .- h2_means),
    markersize=7, markerstrokewidth=1.5,
    color=colors_h2,
    markerstrokecolor=:black,
    xticks=(1:length(trait_labels), trait_labels),
    xlabel="Trait",
    ylabel="Heritability (h²)",
    ylims=(0, min(1.0, maximum(h2_hi) * 1.2)),
    legend=false,
    grid=false,
    framestyle=:box,
    tickfontsize=11,
    labelfontsize=12,
    size=(W_SINGLE, H_SQUARE),
    left_margin=5Plots.mm, bottom_margin=5Plots.mm)

# Add posterior mean labels above each point
for i in eachindex(trait_labels)
    annotate!(fig3, i, h2_hi[i] + 0.02,
        text(@sprintf("%.3f", h2_means[i]), :center, 11, :black))
end

savefig(fig3, joinpath(FIG_DIR, "Fig3_Heritability.pdf"))
savefig(fig3, joinpath(FIG_DIR, "Fig3_Heritability.png"))
println("  Saved Fig3")

# =============================================================================
# 6. FIGURE 4: Genetic correlation posterior from MCMC chains
# =============================================================================
# Estimate posterior distribution of genetic correlation r_g(y1, y2)
# from the MCMC EBV chains: for each sample t, compute cor(EBV_y1^t, EBV_y2^t)
println("Figure 4: Genetic correlation posterior...")

chain_y1_path = joinpath(SAVE_DIR, "MCMC_samples_EBV_y1.txt")
chain_y2_path = joinpath(SAVE_DIR, "MCMC_samples_EBV_y2.txt")

if isfile(chain_y1_path) && isfile(chain_y2_path)
    chains_y1 = Matrix{Float64}(CSV.read(chain_y1_path, DataFrame,
                    delim=',', header=true))
    chains_y2 = Matrix{Float64}(CSV.read(chain_y2_path, DataFrame,
                    delim=',', header=true))

    @assert size(chains_y1) == size(chains_y2) "Chain dimensions differ between traits"
    n_samp = size(chains_y1, 1)

    # Per-sample genetic correlation across all individuals
    r_g_samples = [cor(chains_y1[t, :], chains_y2[t, :]) for t in 1:n_samp]

    r_g_mean = mean(r_g_samples)
    r_g_sd   = std(r_g_samples)
    r_g_lo   = quantile(r_g_samples, 0.025)
    r_g_hi   = quantile(r_g_samples, 0.975)

    @printf("  Genetic correlation r_g(y1,y2): %.3f ± %.3f  95%% CI [%.3f, %.3f]\n",
            r_g_mean, r_g_sd, r_g_lo, r_g_hi)

    fig4 = histogram(r_g_samples,
        bins=40,
        color=COL_GREEN, alpha=0.75,
        linewidth=0,
        xlabel="Genetic correlation r_g(y1, y2)",
        ylabel="Posterior frequency",
        legend=false,
        grid=false,
        framestyle=:box,
        tickfontsize=11,
        labelfontsize=12,
        size=(W_SINGLE, H_SQUARE),
        left_margin=5Plots.mm, bottom_margin=5Plots.mm)

    # Vertical lines: mean and 95% CI
    vline!(fig4, [r_g_mean], color=:black, linewidth=2, linestyle=:solid)
    vline!(fig4, [r_g_lo, r_g_hi], color=:black, linewidth=1, linestyle=:dash)

    # Annotation — estimate y position from sample range
    y_annot = length(r_g_samples) / 40 * 0.8   # approximate bar height
    annotate!(fig4,
        r_g_mean + 0.01,
        y_annot,
        text(@sprintf("%.3f\n[%.3f, %.3f]", r_g_mean, r_g_lo, r_g_hi),
             :left, 8, :black))

    savefig(fig4, joinpath(FIG_DIR, "Fig4_GeneticCorrelation_posterior.pdf"))
    savefig(fig4, joinpath(FIG_DIR, "Fig4_GeneticCorrelation_posterior.png"))
    println("  Saved Fig4")
else
    @warn "Chain files not found — skipping Fig4 (genetic correlation posterior)"
    println("  Expected: $chain_y1_path")
    println("  Expected: $chain_y2_path")
end

# =============================================================================
# 7. SUMMARY TABLE (printed to console)
# =============================================================================
println("\n" * "=" ^ 60)
println("SUMMARY")
println("=" ^ 60)

r_y1 = cor(Float64.(cand_y1.EBV_post_mean), cand_y1.TBV)
r_y2 = cor(Float64.(cand_y2.EBV_post_mean), cand_y2.TBV)

@printf("  Accuracy (r_EBV,TBV) — unphenotyped candidates:\n")
@printf("    y1 (continuous): %.4f\n", r_y1)
@printf("    y2 (binary)    : %.4f\n", r_y2)
@printf("\n  Heritability estimates:\n")
for i in eachindex(trait_labels)
    @printf("    %-6s h² = %.3f ± %.3f  (95%% CI: %.3f – %.3f)\n",
            trait_labels[i], h2_means[i], h2_sds[i], h2_lo[i], h2_hi[i])
end

println("\nAll figures saved to: $FIG_DIR")

# =============================================================================
# 8. FIGURE 5: Posterior EBV densities for selected individuals (y1 only)
#    Shows high-accuracy vs low-accuracy candidates
#    Vertical lines: posterior mean EBV (blue) and TBV (orange dashed)
# =============================================================================
println("\nFigure 5: Individual posterior EBV densities...")

chain_y1_path = joinpath(SAVE_DIR, "MCMC_samples_EBV_y1.txt")

if isfile(chain_y1_path)
    chains_y1_full = CSV.read(chain_y1_path, DataFrame, delim=',', header=true)

    # Identify candidates in chain columns
    chain_ids    = parse.(Int, string.(names(chains_y1_full)))
    cand_id_set  = Set(cand_y1.ID_int)
    cand_chain_cols = findall(id -> id in cand_id_set, chain_ids)

    # Compute per-individual accuracy: |EBV_mean - TBV| as error proxy
    # Use cand_y1 which already has EBV_post_mean and TBV aligned
    cand_y1[!, :post_sd] = cand_y1.postSD_y1

    # Select individuals based on POSTERIOR WIDTH (SD), not prediction error.
    # Narrow posteriors → robust OCS leaves contribution unchanged (like MAP-OCS)
    # Wide posteriors  → robust OCS penalizes, reducing their contribution
    # This directly illustrates the mechanism of the method.
    n_pick = 3
    sorted_by_sd = sort(cand_y1, :post_sd)
    narrow_inds  = sorted_by_sd[1:n_pick, :]               # smallest posterior SD
    wide_inds    = sorted_by_sd[end-n_pick+1:end, :]       # largest posterior SD

    # Build one panel per individual — 6 panels total in 2×3 layout
    panels = []
    colors_good = [COL_BLUE, COL_BLUE, COL_BLUE]   # narrow = blue
    colors_poor = [COL_RED,  COL_RED,  COL_RED]    # wide   = red

    for (group_df, group_colors, group_label) in [
            (narrow_inds, colors_good, "Narrow posterior (low uncertainty)"),
            (wide_inds,   colors_poor, "Wide posterior (high uncertainty)")]

        for (k, row) in enumerate(eachrow(group_df))
            ind_id  = row.ID_int
            ebv_mean = Float64(row.EBV_post_mean)
            tbv_val  = row.TBV
            post_sd  = row.post_sd

            # Get MCMC chain for this individual
            col_idx = findfirst(==(ind_id), chain_ids)
            if isnothing(col_idx)
                continue
            end
            samples = Float64.(chains_y1_full[:, col_idx])

            # KDE via Gaussian kernel on a fine grid
            # x-limits: span both posterior (mean ± 4 SD) AND TBV so both are visible
            post_lo = ebv_mean - 4 * post_sd
            post_hi = ebv_mean + 4 * post_sd
            margin  = 0.1 * (post_hi - post_lo)   # 10% padding
            x_lo    = min(post_lo, tbv_val) - margin
            x_hi    = max(post_hi, tbv_val) + margin
            x_kde   = collect(range(post_lo, post_hi, length=300))

            # Gaussian KDE with bandwidth = 1.06 * σ * n^(-1/5) (Silverman)
            bw    = 1.06 * std(samples) * length(samples)^(-0.2)
            kde_y = [sum(pdf.(Normal(s, bw), xv) for s in samples) / length(samples)
                     for xv in x_kde]

            local p_ind = plot(x_kde, kde_y,
                fill=true, fillalpha=0.35, fillcolor=group_colors[k],
                color=group_colors[k], linewidth=2,
                xlims=(x_lo, x_hi),
                xlabel="EBV",
                ylabel="Density",
                label="",
                grid=false,
                framestyle=:box,
                tickfontsize=11,
                labelfontsize=12)

            # Posterior mean (solid blue) and TBV (dashed orange) — both visible
            vline!(p_ind, [ebv_mean],
                color=COL_BLUE, linewidth=3, linestyle=:solid, label="")
            vline!(p_ind, [tbv_val],
                color=COL_ORANGE, linewidth=3, linestyle=:dash, label="")

            # Annotation: ID + absolute error
            annotate!(p_ind,
                x_lo + 0.04*(x_hi - x_lo),
                maximum(kde_y) * 0.88,
                text(@sprintf("ID %d\nSD=%.2f", ind_id, post_sd),
                     :left, 9, :black))

            push!(panels, p_ind)
        end
    end

    if length(panels) == 6
        # Add legend entries to first panel
        plot!(panels[1], [], [], color=COL_BLUE,   linewidth=2,
              linestyle=:solid, label="Post. mean", legendfontsize=9,
              foreground_color_legend=nothing, legend=:topleft)
        plot!(panels[1], [], [], color=COL_ORANGE, linewidth=2,
              linestyle=:dash,  label="TBV")

        fig5 = plot(panels[1], panels[2], panels[3],
                    panels[4], panels[5], panels[6],
                    layout=(2, 3),
                    size=(W_DOUBLE, round(Int, 120 * MM_TO_PX)),
                    left_margin=5Plots.mm, bottom_margin=5Plots.mm,
                    top_margin=3Plots.mm)

        # Row labels as annotations on leftmost panels
        annotate!(fig5[1], :topright,
            text("Narrow posterior", 10, :black, :bold))
        annotate!(fig5[4], :topright,
            text("Wide posterior", 10, :black, :bold))

        savefig(fig5, joinpath(FIG_DIR, "Fig5_Individual_posteriors_y1.pdf"))
        savefig(fig5, joinpath(FIG_DIR, "Fig5_Individual_posteriors_y1.png"))
        println("  Saved Fig5")
    else
        println("  Could not build Fig5 — $(length(panels)) panels found (expected 6)")
    end
else
    @warn "Chain file not found — skipping Fig5"
    println("  Expected: $chain_y1_path")
end
