# Working Example: MCMC-OCS Uncertainty Analysis

This self-contained example demonstrates the full uncertainty-aware OCS pipeline
on **simulated data** — no access to the original spruce or pine datasets required.

## What it does

1. **Simulates** a forest breeding population (200 individuals, 20 families)
   with genomic relationships, posterior MCMC EBV samples, and MAP EBVs
   that mimic the Norway spruce dataset structure.

2. **Runs MAP-OCS** — the classical approach using point estimates.

3. **Runs MCMC-OCS** — solving OCS separately for each of 500 posterior draws,
   producing a distribution of selection solutions.

4. **Calculates robustness scores** — quantifying each candidate's replaceability
   by measuring gain loss when they are excluded.

5. **Runs Constrained OCS** — re-optimising after excluding the bottom 25%
   of MAP-selected individuals by robustness score.

6. **Generates figures** — EBV vs selection frequency, robustness bar plots,
   and a summary comparison panel.

## Requirements

Julia packages (install once):
```julia
using Pkg
Pkg.add(["JuMP", "COSMO", "CSV", "DataFrames",
         "Statistics", "LinearAlgebra", "Plots", "StatsPlots",
         "DelimitedFiles", "Printf"])
```

## Usage

```julia
# Step 1 — generate simulated data (run once)
include("simulate_example_data.jl")

# Step 2 — run the full pipeline
include("run_example.jl")
```

Runtime is approximately **3–5 minutes** on a standard laptop
(the bottleneck is solving 500 quadratic programmes for MCMC-OCS).

## Expected output

```
RESULTS SUMMARY
───────────────────────────────────────────────────────
Metric                              MAP-OCS  Constrained
───────────────────────────────────────────────────────
Genetic gain (standardised)          ~1.20        ~1.18
Coancestry                           0.020        0.020
Individuals selected                    ~8           ~9
Gain loss                               —          ~1.5%
Mean robustness (selected)           ~0.91        ~0.97
Robustness improvement                  —         ~6–8%
───────────────────────────────────────────────────────
MCMC-OCS EBV-frequency relationship: R² ≈ 0.78 (quadratic)
```

Exact values will vary slightly due to the random seed.

## Scaling to real data

To apply this to your own dataset, replace the data loading in `run_example.jl`:

```julia
# Replace simulated files with your own:
G_raw        = readdlm("your_G_matrix.txt", ',', Float64)
mcmc_samples = readdlm("your_MCMC_EBV_samples.txt", ',', Float64)
map_df       = CSV.read("your_MAP_EBV.csv", DataFrame)
```

File formats mirror JWAS output:
- **G matrix**: CSV with individual IDs as first column, then G values
- **MCMC EBV samples**: rows = individuals, columns = MCMC iterations
- **MAP EBV**: CSV with columns `ID`, `Family`, `MAP_EBV`
