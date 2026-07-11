"""
CVaR-OCS — Norway Spruce and Loblolly Pine (Real Data)
=======================================================

Stochastic Optimum Contribution Selection incorporating MCMC posterior
uncertainty for two forest tree breeding populations.

Species-specific configuration blocks at the top — set SPECIES to
"spruce" or "pine" before running.

No sex constraints (monoecious / clonal systems).
No oracle comparison (real data — TBVs unavailable).

Norway spruce : n=1,218  Θ=0.02  Index=(Hjd17+Htv17−Sprant17)/3
Loblolly pine : n=926    Θ=0.03  Index=(HT6+DBH6+WDN4−GV6)/4

Author: Jon Ahlinder (Skogforsk) / Ahlinder & Waldmann
"""

# Guard
if !@isdefined(FOREST_CVAR_LOADED)
    global FOREST_CVAR_LOADED = true
end

using CSV, DataFrames, Statistics, LinearAlgebra
using JuMP, COSMO
using DelimitedFiles: readdlm
using JLD2, FileIO
using Printf

# =============================================================================
# SPECIES CONFIGURATION — set SPECIES here, everything else is automatic
# =============================================================================

SPECIES = "pine"    # "spruce" or "pine"

if SPECIES == "spruce"

    # ── Norway Spruce ─────────────────────────────────────────────────────────
    SPECIES_LABEL  = "Norway Spruce (Picea abies)"
    N_EXPECTED     = 1218
    THETA          = 0.02

    BASE_DIR       = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\NorwaySpruceData\\"
    SAVE_DIR       = joinpath(BASE_DIR, "Save")
    RESULTS_DIR    = joinpath(BASE_DIR, "results_JWAS_1218_G_adj_Lev17")

    GRM_FILE       = joinpath(SAVE_DIR, "Gmat_1218_spruce_PDF.txt")

    # Posterior mean EBV files (rows = individuals, col 1 = ID, col 2 = EBV)
    EBV_FILES      = Dict(
        "Hjd17"    => joinpath(RESULTS_DIR, "EBV_Hjd17.txt"),
        "Htv17"    => joinpath(RESULTS_DIR, "EBV_Htv17.txt"),
        "Sprant17" => joinpath(RESULTS_DIR, "EBV_Sprant17.txt")
    )

    # MCMC chain files (rows = iterations, cols = individuals)
    MCMC_FILES     = Dict(
        "Hjd17"    => joinpath(RESULTS_DIR, "MCMC_samples_EBV_Hjd17.txt"),
        "Htv17"    => joinpath(RESULTS_DIR, "MCMC_samples_EBV_Htv17.txt"),
        "Sprant17" => joinpath(RESULTS_DIR, "MCMC_samples_EBV_Sprant17.txt")
    )

    # Selection index: (Hjd17 + Htv17 − Sprant17) / 3
    # sign_vec: +1 = maximise, -1 = minimise
    TRAIT_KEYS     = ["Hjd17", "Htv17", "Sprant17"]
    TRAIT_SIGNS    = [+1.0,    +1.0,    -1.0]
    INDEX_DENOM    = 3.0

elseif SPECIES == "pine"

    # ── Loblolly Pine ─────────────────────────────────────────────────────────
    SPECIES_LABEL  = "Loblolly Pine (Pinus taeda)"
    N_EXPECTED     = 926
    THETA          = 0.03

    BASE_DIR       = "C:\\Users\\joah\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData\\"
    SAVE_DIR       = joinpath(BASE_DIR, "Save")
    RESULTS_DIR    = joinpath(BASE_DIR, "Save/results_G_926")   # adjust if different

    GRM_FILE       = joinpath(BASE_DIR, "G_926_MAF001_mis005_rrBLUP_em_JWAS.txt")  # adjust filename

    EBV_FILES      = Dict(
        "HT6"  => joinpath(RESULTS_DIR, "EBV_HT6.txt"),
        "DBH6" => joinpath(RESULTS_DIR, "EBV_DBH6.txt"),
        "WDN4" => joinpath(RESULTS_DIR, "EBV_WDN4.txt"),
        "GV6"  => joinpath(RESULTS_DIR, "EBV_GV6.txt")
    )

    MCMC_FILES     = Dict(
        "HT6"  => joinpath(RESULTS_DIR, "MCMC_samples_EBV_HT6.txt"),
        "DBH6" => joinpath(RESULTS_DIR, "MCMC_samples_EBV_DBH6.txt"),
        "WDN4" => joinpath(RESULTS_DIR, "MCMC_samples_EBV_WDN4.txt"),
        "GV6"  => joinpath(RESULTS_DIR, "MCMC_samples_EBV_GV6.txt")
    )

    # Selection index: (HT6 + DBH6 + WDN4 − GV6) / 4
    TRAIT_KEYS     = ["HT6", "DBH6", "WDN4", "GV6"]
    TRAIT_SIGNS    = [+1.0,  +1.0,   +1.0,   -1.0]
    INDEX_DENOM    = 4.0

else
    error("SPECIES must be \"spruce\" or \"pine\"")
end

# Output directory — species-specific subfolder of SAVE_DIR
OUT_DIR = joinpath(SAVE_DIR, "CVaR_OCS_$(SPECIES)_theta$(replace(string(THETA), "."=>"p"))")
mkpath(OUT_DIR)

# =============================================================================
# SHARED PARAMETERS
# =============================================================================

SELECTION_THRESHOLD  = 1e-4
ALPHA_VALUES         = [0.90, 0.95, 0.99]
MU_VALUES            = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 5.0, 10.0]
COSMO_MAX_ITER       = 30_000
COSMO_EPS            = 1e-5

println("=" ^ 70)
println("CVaR-OCS — Forest Tree Breeding")
println(SPECIES_LABEL)
println("Θ = $THETA  |  Output: $OUT_DIR")
println("=" ^ 70)

# =============================================================================
# HELPER: SELECTION INDEX FROM RAW EBVS
# =============================================================================

"""
    build_index(trait_matrices, signs, denom) -> Vector/Matrix

Combine trait EBV vectors/matrices into a signed, standardised selection index.
Works for both MAP (vector per trait) and MCMC (matrix l×n per trait).
Each trait is standardised before combining to prevent scale dominance.
"""
function standardise_cols(M::Matrix{Float64})
    # standardise each column (individual) independently? No —
    # standardise across individuals (rows of a column vector, or
    # across all values in a scenario row).
    # For a matrix (l×n): standardise within each row (scenario).
    result = similar(M)
    for i in 1:size(M, 1)
        row = M[i, :]
        valid = filter(!isnan, row)
        isempty(valid) && (result[i, :] .= 0.0; continue)
        μ = mean(valid); σ = std(valid)
        result[i, :] = σ > 0 ? (row .- μ) ./ σ : zeros(length(row))
    end
    return result
end

function standardise_vec(v::Vector{Float64})
    valid = filter(!isnan, v)
    isempty(valid) && return zeros(length(v))
    μ = mean(valid); σ = std(valid)
    return σ > 0 ? (v .- μ) ./ σ : zeros(length(v))
end

function build_map_index(ebv_vecs::Vector{Vector{Float64}},
                         signs::Vector{Float64}, denom::Float64)
    n = length(ebv_vecs[1])
    idx = zeros(n)
    for (v, s) in zip(ebv_vecs, signs)
        idx .+= s .* standardise_vec(v)
    end
    idx ./= denom
    replace!(idx, NaN => 0.0)
    return idx
end

function build_mcmc_index(ebv_mats::Vector{Matrix{Float64}},
                          signs::Vector{Float64}, denom::Float64)
    # ebv_mats[k] is (l × n) for trait k
    l, n = size(ebv_mats[1])
    idx = zeros(l, n)
    for (M, s) in zip(ebv_mats, signs)
        idx .+= s .* standardise_cols(M)
    end
    idx ./= denom
    return idx
end

# =============================================================================
# HELPER: EXTRACT EBV MATRIX FROM DATAFRAME (MCMC chains)
# =============================================================================

function extract_ebv_matrix(df::DataFrame)
    col_name = String(names(df)[1])
    first_col = df[:, 1]
    if occursin(r"^(ID|id|iter|Iter|iteration)"i, col_name)
        return Matrix{Float64}(df[:, 2:end])
    end
    if eltype(first_col) <: Integer && all(first_col .== 1:length(first_col))
        return Matrix{Float64}(df[:, 2:end])
    end
    return Matrix{Float64}(df)
end

# =============================================================================
# 1. LOAD GRM
# =============================================================================
println("\n[1] Loading GRM...")

grm_raw = readdlm(GRM_FILE, ',', Float64, '\n', header=false)
grm_ids = Int.(grm_raw[:, 1])
G       = grm_raw[:, 2:end]
n       = size(G, 1)

@assert n == N_EXPECTED "Expected $N_EXPECTED individuals, got $n — check GRM file"
println("  GRM: $n × $n  (diagonal mean = $(round(mean(diag(G)), digits=4)))")

# =============================================================================
# 2. LOAD MAP EBVs AND BUILD SELECTION INDEX
# =============================================================================
println("\n[2] Loading MAP EBVs and building selection index...")

ebv_vecs = Vector{Vector{Float64}}()
for key in TRAIT_KEYS
    df  = CSV.read(EBV_FILES[key], DataFrame, delim=',', missingstring="NA")
    println("  $key columns: $(names(df))")
    # col 2 = posterior mean EBV (col 1 = individual ID)
    push!(ebv_vecs, Float64.(df[:, 2]))
end

g_map = build_map_index(ebv_vecs, TRAIT_SIGNS, INDEX_DENOM)
println("  MAP index range: [$(round(minimum(g_map),digits=3)), $(round(maximum(g_map),digits=3))]")

# =============================================================================
# 3. LOAD MCMC CHAINS AND BUILD SCENARIO MATRIX
# =============================================================================
println("\n[3] Loading MCMC chains...")

cache_file = "tmp" #joinpath(OUT_DIR, "mcmc_index_cache.jld2")

if isfile(cache_file)
    println("  Loading from cache: $cache_file")
    @load cache_file U l_scenarios
    println("  ✓ $l_scenarios scenarios × $n individuals")
else
    ebv_mats = Vector{Matrix{Float64}}()
    for key in TRAIT_KEYS
        println("  Loading $key chain...")
        df  = CSV.read(MCMC_FILES[key], DataFrame, delim=',',
                       missingstring="NA", header=true)
        mat = extract_ebv_matrix(df)
        # Confirm orientation: rows=iterations, cols=individuals
        @assert size(mat, 2) == n "Chain $key has $(size(mat,2)) cols, expected $n"
        push!(ebv_mats, mat)
    end

    l_scenarios = size(ebv_mats[1], 1)
    println("  Building U ($l_scenarios × $n)...")
    U = build_mcmc_index(ebv_mats, TRAIT_SIGNS, INDEX_DENOM)

    @save cache_file U l_scenarios
    println("  ✓ Cache saved: $cache_file")
end

@printf("  Scenario mean index range: [%.3f, %.3f]\n",
        minimum(mean(U, dims=1)), maximum(mean(U, dims=1)))
@printf("  Correlation scenario mean vs MAP index: %.4f\n",
        cor(vec(mean(U, dims=1)), g_map))

# =============================================================================
# 4. COSMO SOLVER FACTORY
# =============================================================================

function make_optimizer()
    return optimizer_with_attributes(
        COSMO.Optimizer,
        "max_iter" => COSMO_MAX_ITER,
        "eps_abs"  => COSMO_EPS,
        "eps_rel"  => COSMO_EPS,
        "verbose"  => false
    )
end

# =============================================================================
# 5. OCS FUNCTIONS (no sex constraints for forest trees)
# =============================================================================

function run_map_ocs(G::Matrix{Float64}, g::Vector{Float64}, theta::Float64)
    n = length(g)
    model = Model(make_optimizer())
    @variable(model, c[1:n] >= 0)
    @constraint(model, sum(c) == 1.0)
    @constraint(model, 0.5 * c' * G * c <= theta)
    @objective(model, Max, dot(g, c))
    optimize!(model)
    status = termination_status(model)
    if status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        cv = value.(c)
        return cv, dot(g, cv), 0.5 * cv' * G * cv, status
    else
        @warn "MAP-OCS did not converge: $status"
        return zeros(n), NaN, NaN, status
    end
end

function run_cvar_ocs(G::Matrix{Float64}, U::Matrix{Float64},
                      theta::Float64, mu::Float64, alpha::Float64)
    l, n = size(U)
    inv_al = 1.0 / ((1.0 - alpha) * l)
    model  = Model(make_optimizer())

    @variable(model, c[1:n] >= 0)
    @variable(model, eta)
    @variable(model, z[1:l] >= 0)

    @constraint(model, sum(c) == 1.0)
    @constraint(model, 0.5 * c' * G * c <= theta)
    for j in 1:l
        @constraint(model, eta - dot(U[j, :], c) - z[j] <= 0)
    end

    g_bar     = vec(mean(U, dims=1))
    cvar_term = eta - inv_al * sum(z)
    @objective(model, Max, dot(g_bar, c) + mu * cvar_term)

    optimize!(model)
    status = termination_status(model)

    if status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        cv  = value.(c)
        etv = value(eta)
        zv  = value.(z)
        return cv, dot(g_bar, cv), etv - inv_al * sum(zv),
               0.5 * cv' * G * cv, status
    else
        @warn "CVaR-OCS did not converge (μ=$mu, α=$alpha): $status"
        return zeros(n), NaN, NaN, NaN, status
    end
end

# =============================================================================
# 6. HELPER: CONTRIBUTION METRICS
# =============================================================================

function contribution_metrics(c::Vector{Float64})
    sel = c[c .> SELECTION_THRESHOLD]
    isempty(sel) && return (n_sel=0, mean_c=NaN, max_c=NaN, max_pct=NaN, gini=NaN)
    sorted = sort(sel)
    ns     = length(sorted)
    ss     = sum(sorted)
    gini   = sum((2i - ns - 1) * sorted[i] for i in 1:ns) / (ns * ss)
    return (n_sel=ns, mean_c=mean(sorted), max_c=maximum(sorted),
            max_pct=maximum(sorted)/ss*100.0, gini=gini)
end

# =============================================================================
# 7. RUN MAP-OCS REFERENCE
# =============================================================================
println("\n[4] Running MAP-OCS reference...")

c_map, gain_map, coanc_map, status_map = run_map_ocs(G, g_map, THETA)
m_map = contribution_metrics(c_map)
sc_map = [dot(c_map, U[j, :]) for j in 1:l_scenarios]

@printf("  Status       : %s\n", string(status_map))
@printf("  MAP gain     : %.6f\n", gain_map)
@printf("  Coancestry   : %.6f  (limit=%.3f)\n", coanc_map, THETA)
@printf("  N selected   : %d\n", m_map.n_sel)
@printf("  Gini         : %.4f\n", m_map.gini)
@printf("  In-sample E[gain] : %.6f\n", mean(sc_map))
@printf("  In-sample VaR95   : %.6f\n", quantile(sc_map, 0.05))
@printf("  In-sample CVaR95  : %.6f\n",
        mean(sc_map[sc_map .< quantile(sc_map, 0.05)]))

# =============================================================================
# 8. CVaR-OCS SWEEP
# =============================================================================
println("\n[5] Sweeping CVaR-OCS over α and μ...")
println("  α: $ALPHA_VALUES")
println("  μ: $MU_VALUES")

frontier_rows = []
solutions     = Dict{Tuple{Float64,Float64}, Vector{Float64}}()

# Store MAP-OCS reference row
push!(frontier_rows, (
    mu=0.0, alpha=NaN, label="MAP-OCS",
    gain_exp=mean(sc_map), cvar_gain=NaN,
    coancestry=coanc_map,
    n_selected=m_map.n_sel, max_contrib=m_map.max_c,
    max_pct=m_map.max_pct, gini=m_map.gini,
    var95=quantile(sc_map, 0.05),
    cvar95_eval=mean(sc_map[sc_map .< quantile(sc_map, 0.05)]),
    status=string(status_map)
))

for alpha in ALPHA_VALUES
    for mu in MU_VALUES
        label = @sprintf("CVaR(α=%.2f,μ=%.2f)", alpha, mu)
        print("  $label ...")

        c, gain_exp, cvar_gain, coanc, status =
            run_cvar_ocs(G, U, THETA, mu, alpha)

        if status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
            m    = contribution_metrics(c)
            sc   = [dot(c, U[j, :]) for j in 1:l_scenarios]
            var95  = quantile(sc, 0.05)
            cvar95 = mean(sc[sc .< var95])

            push!(frontier_rows, (
                mu=mu, alpha=alpha, label=label,
                gain_exp=gain_exp, cvar_gain=cvar_gain,
                coancestry=coanc,
                n_selected=m.n_sel, max_contrib=m.max_c,
                max_pct=m.max_pct, gini=m.gini,
                var95=var95, cvar95_eval=cvar95,
                status=string(status)
            ))
            solutions[(mu, alpha)] = c
            @printf("  E[gain]=%.4f  CVaR95=%.4f  n=%d  gini=%.3f\n",
                    gain_exp, cvar95, m.n_sel, m.gini)
        else
            @printf("  FAILED (%s)\n", string(status))
        end
    end
end

# =============================================================================
# 9. SAVE FRONTIER AND CONTRIBUTION VECTORS
# =============================================================================
println("\n[6] Saving results...")

frontier_df   = DataFrame(frontier_rows)
frontier_file = joinpath(OUT_DIR, "cvar_ocs_frontier_$(SPECIES).csv")
CSV.write(frontier_file, frontier_df)
println("  Frontier → $frontier_file")

# Wide contribution matrix: rows = models, cols = individual IDs
contrib_df = DataFrame(label=String[], mu=Float64[], alpha=Float64[])
for id in grm_ids
    contrib_df[!, "ID_$id"] = Float64[]
end

# MAP-OCS row
push!(contrib_df, vcat(["MAP-OCS", 0.0, NaN], c_map))

for ((mu, alpha), c) in solutions
    lbl = @sprintf("CVaR_a%.2f_mu%.2f", alpha, mu)
    push!(contrib_df, vcat([lbl, mu, alpha], c))
end

contrib_file = joinpath(OUT_DIR, "cvar_ocs_solutions_$(SPECIES).csv")
CSV.write(contrib_file, contrib_df)
println("  Contributions → $contrib_file")

# =============================================================================
# 10. SUMMARY COMPARISON
# =============================================================================
println("\n" * "=" ^ 70)
println("COMPARISON: MAP-OCS vs best CVaR-OCS per α")
println("=" ^ 70)

function best_for_alpha(alpha_val)
    sub = frontier_df[abs.(frontier_df.alpha .- alpha_val) .< 0.001, :]
    isempty(sub) && return nothing
    return sub[argmax(sub.cvar95_eval), :]
end

@printf("\n  %-32s  %8s  %8s  %6s  %6s\n",
        "Model", "E[gain]", "CVaR95", "N_sel", "Gini")
println("  " * "─" ^ 65)

map_r = frontier_df[frontier_df.label .== "MAP-OCS", :]
@printf("  %-32s  %8.4f  %8.4f  %6d  %6.4f\n",
        "MAP-OCS", map_r.gain_exp[1], map_r.cvar95_eval[1],
        map_r.n_selected[1], map_r.gini[1])

for alpha_val in ALPHA_VALUES
    br = best_for_alpha(alpha_val)
    br === nothing && continue
    delta_cvar = (br.cvar95_eval - map_r.cvar95_eval[1]) /
                 abs(map_r.cvar95_eval[1]) * 100
    delta_gain = (br.gain_exp - map_r.gain_exp[1]) /
                 abs(map_r.gain_exp[1]) * 100
    @printf("  %-32s  %8.4f  %8.4f  %6d  %6.4f  (ΔCVaR=%+.1f%%  ΔE[gain]=%+.1f%%)\n",
            br.label, br.gain_exp, br.cvar95_eval,
            br.n_selected, br.gini, delta_cvar, delta_gain)
end

println("\n" * "=" ^ 70)
println("CVaR-OCS COMPLETE — $SPECIES_LABEL")
println("=" ^ 70)
@printf("Output directory: %s\n", OUT_DIR)
