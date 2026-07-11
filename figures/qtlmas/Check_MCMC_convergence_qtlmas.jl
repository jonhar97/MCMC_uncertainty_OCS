using DataFrames, CSV, Statistics
using Plots
using Printf
using MCMCDiagnosticTools   # Pkg.add("MCMCDiagnosticTools") if missing

# ============================================================================
# PATHS — QTL-MAS 2010
# ============================================================================

BASE_PATH  = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\QTLMAS"
SAVE_PATH  = joinpath(BASE_PATH, "Save6")
FIG_PATH   = joinpath(BASE_PATH, "Figures")
mkpath(FIG_PATH)

# Files sit directly in Save5, no subfolder — point straight at it
ACTUAL_DIR_QTLMAS = SAVE_PATH
println("Diagnosing chain in: $ACTUAL_DIR_QTLMAS")

# ============================================================================
# HELPER FUNCTIONS (unchanged from spruce script)
# ============================================================================

function load_jwas_chain(path::String)
    df = CSV.read(path, DataFrame)
    numeric_cols = [c for c in names(df) if eltype(df[!, c]) <: Union{Real, Missing}]
    if isempty(numeric_cols)
        error("No numeric columns found in $path\nColumns: $(names(df))\nTypes: $(eltype.(eachcol(df)))")
    end
    return Matrix{Float64}(df[!, numeric_cols])
end

function report_ess(chain::Matrix{Float64};
                    param_names::Vector{String}=String[],
                    min_ess::Int=200)
    n_s, n_p = size(chain)
    ess_vals = [MCMCDiagnosticTools.ess(chain[:, i:i])[1] for i in 1:n_p]
    println(@sprintf("  %-35s  %8s  %6s", "Parameter", "ESS", "ESS/n"))
    println("  " * "-"^54)
    any_low = false
    for i in 1:n_p
        name = i <= length(param_names) ? param_names[i] : "param_$i"
        e    = ess_vals[i]
        flag = e < min_ess ? "  <- LOW" : ""
        if e < min_ess; any_low = true; end
        println(@sprintf("  %-35s  %8.1f  %6.3f%s", name, e, e/n_s, flag))
    end
    any_low ? println("\n  WARNING: Some ESS < $min_ess") :
              println("\n  OK: All ESS >= $min_ess")
    return ess_vals
end

function plot_convergence(chain::Matrix{Float64},
                          param_names::Vector{String},
                          title_str::String, outpath::String;
                          max_params::Int=8)
    n_p = min(size(chain, 2), max_params)
    panels = []
    for i in 1:n_p
        s    = chain[:, i]
        name = i <= length(param_names) ? param_names[i] : "param_$i"
        pt = plot(s, lw=0.5, color=:steelblue, label="",
                  xlabel="Sample", ylabel=name,
                  title="Trace: $name", titlefontsize=8, guidefontsize=7)
        plot!(pt, cumsum(s) ./ (1:length(s)), lw=1.5, color=:red, label="Running mean")
        pd = histogram(s, normalize=:pdf, color=:lightblue, label="",
                       xlabel=name, ylabel="Density",
                       title="Posterior: $name", titlefontsize=8, guidefontsize=7)
        push!(panels, pt); push!(panels, pd)
    end
    fig = plot(panels..., layout=(n_p, 2), size=(900, 240*n_p),
               plot_title=title_str, plot_titlefontsize=9, margin=3Plots.mm)
    savefig(fig, outpath)
    println("  Saved: $outpath")
    return fig
end

# ============================================================================
# PART 2: ESS + TRACE PLOTS
# ============================================================================

for (save_dir, label, trait_names) in [
        (ACTUAL_DIR_QTLMAS, "QTLMAS900", ["y1", "y2"])
    ]

    println("\n--- $label ($(save_dir)) ---")
    if !isdir(save_dir)
        println("  Directory not found."); continue
    end

    herit_path = ""
    for c in ["MCMC_samples_heritability.txt", "heritability.txt",
              "heritability", "heritability.csv", "Heritability"]
        isfile(joinpath(save_dir, c)) && (herit_path = joinpath(save_dir, c); break)
    end

    if herit_path != ""
        h_chain = load_jwas_chain(herit_path)
        names_h = trait_names[1:min(length(trait_names), size(h_chain, 2))]
        println("\n  ESS — $label heritabilities ($(size(h_chain,1)) samples):")
        report_ess(h_chain, param_names=names_h)
        plot_convergence(h_chain, names_h,
                         "Convergence: $label heritabilities",
                         joinpath(FIG_PATH, "convergence_$(label)_heritability.pdf"))
    else
        println("  heritability chain not found — check filenames listed above")
    end

    var_path = ""
    for c in ["MCMC_samples_genetic_variance.txt", "genetic_variance.txt",
              "genetic_variance", "genetic_variance.csv"]
        isfile(joinpath(save_dir, c)) && (var_path = joinpath(save_dir, c); break)
    end

    if var_path != ""
        v_chain = load_jwas_chain(var_path)
        n_t     = length(trait_names)
        names_v = vcat(
            ["s2_g($(t))" for t in trait_names],
            ["cov_g($(trait_names[i]),$(trait_names[j]))"
             for i in 1:n_t for j in (i+1):n_t]
        )[1:min(end, size(v_chain, 2))]
        println("\n  ESS — $label genetic variances:")
        report_ess(v_chain, param_names=names_v)
        plot_convergence(v_chain, names_v,
                         "Convergence: $label genetic variances",
                         joinpath(FIG_PATH, "convergence_$(label)_genetic_variance.pdf"))
    end
end