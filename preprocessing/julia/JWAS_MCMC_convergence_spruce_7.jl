"""
JWAS MCMC: Correct Settings + Convergence Diagnostics
Norway Spruce — G_1218 and H_5525
======================================================

ROOT CAUSE OF ALL PREVIOUS ERRORS:
  JWAS resolves the term "genotypes" in the model equation string by looking
  for a variable literally named `genotypes` in the calling scope at runtime.
  It is NOT passed to runMCMC — it must exist as a global named `genotypes`.
  Your original script worked interactively because that global persisted in
  the REPL between cells. From include() it was absent, causing every error.

  Fix: assign get_genotypes(...) to a variable named exactly `genotypes`
  before calling build_model / runMCMC. Run one model at a time so the
  global is not clobbered between the two models.

  add_genotypes() is NOT used — it only handles raw SNP dosage files (0/1/2),
  not pre-computed GRM files.

CHAIN ARITHMETIC:
  chain_length             = 52000   (was 1000 — too short for convergence)
  burnin                   = 2000
  output_samples_frequency = 10      (= thinning; no `thinning` kwarg in JWAS)
  samples saved = (52000 - 2000) / 10 = 5000
"""

using JWAS, DataFrames, CSV, Statistics
using Plots
using Printf
using MCMCDiagnosticTools   # Pkg.add("MCMCDiagnosticTools") if missing

# ============================================================================
# CHAIN SETTINGS
# ============================================================================

CHAIN_LENGTH = 52000
BURNIN       = 2000
OUTPUT_FREQ  = 10
N_SAMPLES    = div(CHAIN_LENGTH - BURNIN, OUTPUT_FREQ)   # 5000

println("Chain settings:")
println("  chain_length             = $CHAIN_LENGTH")
println("  burnin                   = $BURNIN")
println("  output_samples_frequency = $OUTPUT_FREQ  (= thinning)")
println("  posterior samples saved  = $N_SAMPLES")

# ============================================================================
# PATHS
# ============================================================================

BASE_PATH  = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData"
SAVE_PATH  = joinpath(BASE_PATH, "Save")
FIG_PATH   = joinpath(BASE_PATH, "Figures")
mkpath(FIG_PATH)

# Output folder names — JWAS appends a number if folder exists; we find the
# actual folder used via startswith() after the run.
SAVE_DIR_1218 = joinpath(SAVE_PATH, "MCMC_G1218")
SAVE_DIR_5525 = joinpath(SAVE_PATH, "MCMC_H5525")

# ============================================================================
# HELPER: find actual JWAS output folder (handles auto-incrementing)
# ============================================================================

function find_jwas_output(save_path::String, prefix::String)
    matches = filter(d -> startswith(d, prefix), readdir(save_path))
    isempty(matches) && error("No folder starting with '$prefix' found in $save_path")
    # Sort and take last (highest number = most recent)
    actual = joinpath(save_path, sort(matches)[end])
    println("  Output folder: $actual")
    return actual
end

# ============================================================================
# PART 1A: G-matrix model (n=1218)
# ============================================================================

RUN_G1218 = false   # set false once chain files exist

if RUN_G1218
    println("\n" * "="^70)
    println("RUNNING G-matrix model (n=1218)")
    println("="^70)

    phenofile_1218 = joinpath(SAVE_PATH, "JWAS_phenotypes_1218.txt")
    genofile_1218  = joinpath(SAVE_PATH, "JWAS_G_1218_tuned.txt")

    phenotypes_1218 = CSV.read(phenofile_1218, DataFrame,
                               delim=',', header=true,
                               missingstring=["NA"])

    # KEY: variable MUST be named `genotypes` — JWAS resolves this name from
    # the model equation string at runtime via the calling scope.
    global genotypes = get_genotypes(genofile_1218, separator=',',
                                     method="GBLUP", header=false, rowID=true)

    model_eq_1218 = "Hjd17    = intercept + Trial + Trial_Ruta + genotypes
                     Htv17    = intercept + Trial + Trial_Ruta + genotypes
                     Sprant17 = intercept + Trial + Trial_Ruta + genotypes"

    model_1218 = build_model(model_eq_1218)
    set_random(model_1218, "Trial_Ruta")

    out_1218 = runMCMC(
        model_1218,
        phenotypes_1218;
        chain_length             = CHAIN_LENGTH,
        burnin                   = BURNIN,
        output_samples_frequency = OUTPUT_FREQ,
        output_folder            = SAVE_DIR_1218,
        outputEBV                = true,
        output_heritability      = true
    )

    println("✓ G1218 complete.  Heritabilities:")
    println(out_1218["heritability"])
    actual_1218 = find_jwas_output(SAVE_PATH, "MCMC_G1218")
    CSV.write(joinpath(actual_1218, "heritability_summary.csv"),     out_1218["heritability"])
    CSV.write(joinpath(actual_1218, "genetic_variance_summary.csv"), out_1218["genetic_variance"])
    println("Files written:")
    for f in readdir(actual_1218); println("  $f"); end
    global ACTUAL_DIR_1218 = actual_1218
end

# ============================================================================
# PART 1B: H-matrix model (n=5525)
# ============================================================================

RUN_H5525 = false

if RUN_H5525
    println("\n" * "="^70)
    println("RUNNING H-matrix model (n=5525)")
    println("="^70)

    phenofile_5525 = joinpath(SAVE_PATH, "phenotypes_5525_spruce_Horder_v3.txt")
    genofile_5525  = joinpath(SAVE_PATH, "Hmat_5525_spruce_tau_1_omega_1_PDF.txt")

    phenotypes_5525 = CSV.read(phenofile_5525, DataFrame,
                               delim=',', header=true,
                               missingstring=["NA"])
    phenotypes_5525.Lev17 = replace(phenotypes_5525.Lev17, 1 => 2, 0 => 1)

    # Overwrite global `genotypes` with H5525 version
    global genotypes = get_genotypes(genofile_5525, separator=',',
                                     method="GBLUP", header=false)
    # Fix scientific notation IDs (e.g. "1.23e4" -> "12300")
    genotypes.obsID = [string(Int64(parse(Float64, id))) for id in genotypes.obsID]

    model_eq_5525 = "Hjd17    = intercept + Trial + Trial_Ruta + genotypes
                     Htv17    = intercept + Trial + Trial_Ruta + genotypes
                     Sprant17 = intercept + Trial + Trial_Ruta + genotypes
                     Lev17    = intercept + Trial + Trial_Ruta + genotypes"

    model_5525 = build_model(model_eq_5525, categorical_trait=["Lev17"])
    set_random(model_5525, "Trial_Ruta")

    out_5525 = runMCMC(
        model_5525,
        phenotypes_5525;
        chain_length             = CHAIN_LENGTH,
        burnin                   = BURNIN,
        output_samples_frequency = OUTPUT_FREQ,
        output_folder            = SAVE_DIR_5525,
        outputEBV                = true,
        output_heritability      = true
    )

    println("✓ H5525 complete.  Heritabilities:")
    println(out_5525["heritability"])
    actual_5525 = find_jwas_output(SAVE_PATH, "MCMC_H5525")
    CSV.write(joinpath(actual_5525, "heritability_summary.csv"),     out_5525["heritability"])
    CSV.write(joinpath(actual_5525, "genetic_variance_summary.csv"), out_5525["genetic_variance"])
    println("Files written:")
    for f in readdir(actual_5525); println("  $f"); end
    global ACTUAL_DIR_5525 = actual_5525
end

# ============================================================================
# PART 2: CONVERGENCE DIAGNOSTICS
# ============================================================================
# Both models ran successfully — set RUN_G1218 = RUN_H5525 = false above
# and re-include this script to run diagnostics only, pointing to existing
# folders. Or set these manually if needed:
#   ACTUAL_DIR_1218 = joinpath(SAVE_PATH, "MCMC_G12183")   # adjust number
#   ACTUAL_DIR_5525 = joinpath(SAVE_PATH, "MCMC_H55251")   # adjust number

# Fallback: if models were not run in this session, find most recent folders
if !@isdefined(ACTUAL_DIR_1218)
    matches = sort(filter(d -> startswith(d, "MCMC_G1218"), readdir(SAVE_PATH)))
    isempty(matches) && error("No MCMC_G1218* folder found in $SAVE_PATH")
    global ACTUAL_DIR_1218 = joinpath(SAVE_PATH, matches[end])
    println("Using existing G1218 folder: $ACTUAL_DIR_1218")
end

if !@isdefined(ACTUAL_DIR_5525)
    matches = sort(filter(d -> startswith(d, "MCMC_H5525"), readdir(SAVE_PATH)))
    isempty(matches) && error("No MCMC_H5525* folder found in $SAVE_PATH")
    global ACTUAL_DIR_5525 = joinpath(SAVE_PATH, matches[end])
    println("Using existing H5525 folder: $ACTUAL_DIR_5525")
end

println("\n" * "="^70)
println("PART 2: CONVERGENCE DIAGNOSTICS")
println("="^70)

function load_jwas_chain(path::String)
    df = CSV.read(path, DataFrame)
    # Keep only columns that are numeric — JWAS files sometimes include a
    # string ID/iteration column as the first column
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
    # Compute ESS column by column to avoid scalar-vs-vector return type issues
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

for (save_dir, label, trait_names) in [
        (ACTUAL_DIR_1218, "G1218", ["Hjd17", "Htv17", "Sprant17"]),
        (ACTUAL_DIR_5525, "H5525", ["Hjd17", "Htv17", "Sprant17", "Lev17"])
    ]

    println("\n--- $label ($(save_dir)) ---")
    if !isdir(save_dir)
        println("  Directory not found — run Part 1 first."); continue
    end

    println("  Files present:")
    for f in readdir(save_dir); println("    $f"); end

    # Heritability chain — JWAS writes MCMC samples to MCMC_samples_heritability.txt
    # and posterior summary to heritability.txt — we want the samples file
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

    # Genetic variance chain
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

# ============================================================================
# PART 3: METHODS TEXT
# ============================================================================

println("\n" * "="^70)
println("PART 3: METHODS TEXT FOR MANUSCRIPT")
println("="^70)
println("""
Add to the MCMC methods paragraph:

  "MCMC chain convergence was assessed by visual inspection of trace plots
  and by computing the effective sample size (ESS) for all heritability and
  genetic variance parameters using the MCMCDiagnosticTools.jl package.
  A total of $N_SAMPLES posterior samples were retained after discarding
  a burn-in of $BURNIN iterations, with one sample saved per $OUTPUT_FREQ
  iterations (total chain length $CHAIN_LENGTH). ESS values exceeded [X]
  for all heritability parameters (Supplementary Figure SX), confirming
  adequate posterior mixing. The GBLUP formulation involves substantially
  fewer variance parameters than whole-genome marker-effect models, and
  the resulting chain autocorrelation is accordingly lower, consistent with
  previous reports for GBLUP analyses of comparable size
  (Sorensen & Gianola 2002)."

Fill in [X] with the minimum ESS from the output above.
Supplementary figure = convergence_*_heritability.pdf files.
""")
