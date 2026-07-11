# CVaR-OCS: Uncertainty-Aware Optimum Contribution Selection

[![Julia](https://img.shields.io/badge/Julia-1.9+-purple.svg)](https://julialang.org)
[![R](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code repository for the manuscript:

> **Uncertainty-aware breeding decisions: MCMC-based optimum contribution selection increases breeding decision robustness**
> Jon Ahlinder¹ & Patrik Waldmann²
> ¹ Department of Tree Breeding, Skogforsk (The Forestry Research Institute of Sweden)
> ² Mathematical Sciences, University of Oulu, Finland
> *GENETICS* (in press, 2026)

---

## Overview

Optimum Contribution Selection (OCS) balances genetic gain against inbreeding by optimizing parental contributions to the next generation. Classical implementations rely on point estimates (MAP) of breeding values (EBVs) and discard the uncertainty inherent in genetic evaluations.

This repository introduces and evaluates **CVaR-OCS**, a single-solve formulation that incorporates the full MCMC posterior distribution of EBVs directly into the OCS objective via **Conditional Value at Risk (CVaR)** — a coherent risk measure from financial portfolio theory. Rather than treating breeding values as known, CVaR-OCS simultaneously maximizes expected genetic gain and protects against worst-case outcomes driven by EBV uncertainty. A complementary **MCMC-OCS** procedure (re-solving OCS across posterior draws) is used as a diagnostic/characterization tool, yielding individual robustness scores and selection-stability metrics.

The method is evaluated on three datasets:

- **QTL-MAS 2010** simulated benchmark (n = 3,226, 900 unphenotyped selection candidates) — true breeding values are known, enabling comparison against an oracle solution
- **Norway spruce** (*Picea abies*; n = 5,525 H-matrix / n = 1,218 G-matrix)
- **Loblolly pine** (*Pinus taeda*; n = 926)

Headline results: on QTL-MAS, point-estimate MAP-OCS recovered only 78% of the oracle's achievable gain, while CVaR-OCS matched MAP-OCS's expected gain and improved tail-gain security (CVaR₉₅ +0.68%). In Norway spruce, the recommended CVaR-OCS operating point improved tail-gain security by 6.60% and broadened the selection base from 145 to 159 individuals at a genetic gain cost of only 0.70%. Robustness scores further showed that 25 MAP-OCS selections in spruce were unstable across the posterior — but simply excluding them post hoc did not improve tail-gain security, motivating the principled CVaR-OCS approach over ad-hoc exclusion.

---

## Repository Structure

```
MCMC_uncertainty_OCS/
│
├── README.md
├── .gitignore
│
├── data/
│   ├── norway_spruce/             # Anonymized spruce data (IDs recoded, cross-dataset consistent)
│   │   ├── JWAS_phenotypes_1218_anon.txt
│   │   ├── JWAS_G_1218_tuned2_anon.txt      # G matrix, n=1,218
│   │   ├── JWAS_A_1218_anon.txt             # A matrix, n=1,218
│   │   ├── phenotypes_5525_spruce_anon.txt
│   │   ├── Hmat_5525_spruce_anon.txt.gz     # H matrix, n=5,525 (gzip-compressed, see below)
│   │   ├── JWAS_A_5525_anon.txt.gz          # A matrix, n=5,525 (gzip-compressed, see below)
│   │   ├── EBV_Hjd17.txt, EBV_Hjd7.txt, EBV_Htv17.txt,
│   │   │   EBV_Sprant17.txt, EBV_Lev17.txt  # Posterior-mean EBVs, n=1,218 subset (5 traits)
│   │   └── MCMC_samples_EBV_*.txt           # Full posterior chains — NOT tracked in git (~115MB);
│   │                                          # regenerate via preprocessing/julia/, see Data Availability
│   ├── loblolly_pine/
│   │   ├── EBV_DBH6.txt, EBV_GV6.txt, EBV_HT6.txt, EBV_WDN4.txt  # Posterior-mean EBVs (4 traits)
│   │   └── MCMC_samples_EBV_*.txt           # Full posterior chains — NOT tracked in git (~350MB)
│   └── qtlmas/
│       ├── EBV_y1.txt, EBV_y2.txt           # Posterior-mean EBVs, continuous (y1) + binary (y2) traits
│       └── MCMC_samples_EBV_*.txt           # Full posterior chains — NOT tracked in git (~620MB)
│
├── preprocessing/                # Data preparation and MCMC-GBLUP fitting (run before src/)
│   ├── R/
│   │   ├── DataWrangling_spruce.Rmd         # Phenotype setup, G/A/H matrix construction (spruce)
│   │   ├── Update_data_spruce.Rmd           # G matrix tuning, phenotype/matrix alignment (spruce)
│   │   └── Amatrix_taeda.Rmd                # A matrix construction (Loblolly pine)
│   └── julia/
│       ├── JWAS_MCMC_convergence_spruce_7.jl # Fits G_1218 and H_5525 GBLUP models (corrected chain
│       │                                      # length: 52,000 iter / 2,000 burn-in) + convergence diagnostics
│       ├── JWAS_GBLUP_spruce_5022_A.jl       # JWAS MCMC-GBLUP, spruce (A matrix)
│       ├── JWASGBLUP_QTLMAS2010_9.jl         # Bivariate JWAS MCMC-GBLUP, QTL-MAS 2010
│       ├── JWAS_GBLUP_taeda_926_G.jl         # JWAS MCMC-GBLUP, pine (G matrix)
│       └── JWAS_GBLUP_taeda_926_A.jl         # JWAS MCMC-GBLUP, pine (A matrix)
│
├── src/                          # Core CVaR-OCS / MCMC-OCS pipeline
│   ├── cvar_ocs_forest_trees.jl              # CVaR-OCS, unified spruce+pine (set SPECIES = "spruce"|"pine")
│   ├── forest_robustness_analysis_2.jl       # MCMC robustness analysis, unified (saved run: SPECIES="spruce")
│   ├── forest_robustness_analysis_pine.jl    # Same script, saved run: SPECIES="pine"
│   ├── benchmark_ocs_timing_4.jl             # Wall-clock benchmarking, MAP-OCS vs CVaR-OCS vs robustness (Table S9)
│   │
│   ├── norway_spruce/
│   │   └── MCMC_uncertainty_pabi1218_upd5.jl # MCMC-OCS diagnostic/characterization, overlap analysis (spruce, n=1,218)
│   │
│   ├── loblolly_pine/
│   │   └── MCMC_uncertainty_taeda.jl         # MCMC-OCS diagnostic/characterization, overlap analysis (pine)
│   │
│   └── qtlmas/
│       ├── cvar_ocs_qtlmas2010_5.jl          # CVaR-OCS + MAP-OCS + oracle comparison, QTL-MAS
│       ├── MCMC_diagnostics_QTLMAS.jl        # Overlap/selection-frequency diagnostics (run after cvar_ocs_qtlmas2010_5.jl)
│       └── QTLMAS_mcmc_robustness_2.jl       # MCMC robustness analysis, QTL-MAS candidates
│
├── figures/                      # Publication figure generation
│   ├── cvar_frontier_elbow.jl                # Efficiency-frontier elbow detection + figure (SPECIES = "spruce"|"qtlmas")
│   ├── cvar_frontier_elbow_pine.jl           # Same analysis, pine
│   │
│   ├── norway_spruce/
│   │   └── figure_cvar_spruce_composite_1.jl # 4-panel composite: frontier, marginal efficiency, EBV vs robustness, gain distribution
│   │
│   ├── loblolly_pine/
│   │   ├── figure_cvar_pine_composite.jl     # 4-panel composite (pine)
│   │   └── Check_MCMC_convergence_pine.jl    # MCMC convergence diagnostics (ESS, trace plots)
│   │
│   └── qtlmas/
│       ├── figure_cvar_qtlmas_composite_2.jl # 4-panel composite: frontier, EBV/TBV vs contribution, gain distribution
│       ├── plot_QTLMAS2010_results_JWAS.jl   # GBLUP diagnostics: EBV vs TBV, heritability/genetic correlation posteriors
│       └── Check_MCMC_convergence_qtlmas.jl  # MCMC convergence diagnostics (ESS, trace plots)
│
├── utils/                        # Helper scripts and secondary analyses
│   ├── add_family_column_taeda.jl
│   ├── add_quartiles_taeda.jl
│   ├── identify_families_taeda.jl
│   ├── overlap_analysis_GP.jl
│   ├── analyze_norway_spruce_ebv_selection.jl
│   └── load_and_analyze_norway_spruce.jl
│
└── example/                      # Self-contained working example (simulated data)
    ├── README.md
    ├── simulate_example_data.jl
    └── run_example.jl
```

---

## Analysis Workflow

```
Stage 1: preprocessing/R/       →  Construct A, G, H matrices; wrangle phenotypes
Stage 2: preprocessing/julia/   →  Fit MCMC-GBLUP in JWAS; save posterior EBV samples
Stage 3: src/                   →  Run CVaR-OCS, MCMC-OCS diagnostics, robustness scores
Stage 4: figures/               →  Elbow detection and publication figures from saved results
```

---

## Key Methods

### CVaR-OCS

CVaR-OCS extends classical OCS to a stochastic program over `l` MCMC posterior draws `g⁽ʲ⁾` of the breeding values, using a sample-average approximation of Conditional Value at Risk (Rockafellar & Uryasev 2000, 2002):

```
maximize_{c, η, z}   (1/l) Σⱼ c'g⁽ʲ⁾ + μη − 1/((1−α)l) Σⱼ zⱼ

subject to   zⱼ ≥ η − c'g⁽ʲ⁾,   j = 1,...,l
             zⱼ ≥ 0,             j = 1,...,l
             ½ c'Σc ≤ Θ                          (coancestry constraint)
             c's = 0.5, c'd = 0.5                (sex balance, where applicable)
             c'1 = 1
             0 ≤ c ≤ m
```

`α` is the CVaR confidence level, `μ ≥ 0` is the tail-risk penalty weight, and `η` recovers the Value-at-Risk at level `α`. Setting `μ = 0` recovers ordinary MAP-OCS on the posterior-mean EBVs.

### Operating point selection (efficiency-frontier elbow)

For each `α`, CVaR-OCS is solved across a grid of `μ` values, tracing a frontier of genetic-gain loss vs. CVaR₉₅ improvement relative to MAP-OCS. The recommended operating point is the geometric elbow of this frontier (chord-from-first-to-last-point method) — the point beyond which additional tail-gain protection costs disproportionately more expected gain. Each `α`/dataset combination has its own independently derived elbow (`figures/cvar_frontier_elbow*.jl`).

### MCMC-OCS robustness scores

As a diagnostic complementary to CVaR-OCS, ordinary MAP-OCS is re-solved independently at each of the `l` posterior draws, producing a distribution of selection solutions. Individual robustness is summarized via selection frequency, Jaccard similarity and Spearman rank correlation between the MAP solution and the per-draw MCMC solutions (`src/*/`, `src/qtlmas/QTLMAS_mcmc_robustness_2.jl`, `MCMC_diagnostics_QTLMAS.jl`). This reveals which MAP-OCS selections are stable vs. driven by estimation noise, without itself constituting a selection decision rule.

---

## Dependencies

### Julia packages

```julia
using Pkg
Pkg.add([
    "JWAS",               # Bayesian mixed model MCMC (GBLUP)
    "JuMP", "COSMO",       # Optimization modeling + QP/SOCP solver
    "DataFrames", "CSV",
    "Statistics", "StatsBase", "LinearAlgebra", "Distributions",
    "HypothesisTests",     # Kendall/Spearman correlation
    "MCMCDiagnosticTools", # ESS / convergence diagnostics
    "Plots", "StatsPlots", "KernelDensity", "LaTeXStrings", "Measures", "Colors", "PlotThemes",
    "AbstractGPs", "KernelFunctions", "Optim",   # Gaussian process regression
    "DelimitedFiles", "JLD2", "FileIO",
    "ProgressMeter", "Distributed", "DataStructures", "Random",
    "Printf", "Dates"
])
```

### R packages

```r
install.packages(c(
    "dplyr", "ggplot2", "nadiv",      # Data wrangling and A matrix
    "AGHmatrix",                       # G matrix construction
    "pheatmap",                        # Matrix visualisation
    "corpcor",                         # Positive-definiteness correction
    "vegan"                            # Multivariate statistics
))
```

Tested with Julia 1.10 and R 4.3.

---

## Data Availability

**Norway spruce** phenotypic and genomic relationship data, with individual IDs recoded and consistent across the n=1,218 and n=5,525 files, are provided in `data/norway_spruce/`. The private ID-mapping key used to anonymize these files is intentionally **not** included in this repository (see `.gitignore`) and is retained locally.

The two dense n=5,525 relationship matrices (`Hmat_5525_spruce_anon.txt`, 455 MB; `JWAS_A_5525_anon.txt`, 68 MB) exceed GitHub's file size limits as plain text, so they are stored gzip-compressed (`.txt.gz`, 79 MB and 2.2 MB respectively — the H matrix in particular compresses well due to repeated values within family blocks). Decompress before running the preprocessing/analysis scripts:

```bash
gunzip -k data/norway_spruce/Hmat_5525_spruce_anon.txt.gz
gunzip -k data/norway_spruce/JWAS_A_5525_anon.txt.gz
```

(`-k` keeps the compressed copy alongside the extracted `.txt`.)

Posterior-mean ("MAP") EBV files are provided for all three datasets under `data/norway_spruce/`, `data/loblolly_pine/`, and `data/qtlmas/` and are small enough to track directly. The full MCMC posterior EBV chains (`MCMC_samples_EBV_*.txt`, used to build the CVaR-OCS scenario matrices) are **not currently tracked in this repository** — combined they total roughly 1.1GB and don't compress well enough to fit GitHub's limits. Regenerate them by re-running the relevant `preprocessing/julia/` script, or check back here for an external archive link (e.g. Zenodo) if one is added in a future update.

**Loblolly pine** data are derived from publicly available NCBI SRA records as described in the manuscript and are not bundled here.

**QTL-MAS 2010** is a public simulated benchmark (Szydłowski & Paczyńska 2011); raw input files (`tbv.txt`, `phenotypes.txt`, genotype file) are not yet included in this repository — add them under a `data/qtlmas/` folder if you want the QTL-MAS pipeline to be runnable end-to-end from this repo, or point `preprocessing/julia/JWASGBLUP_QTLMAS2010_9.jl` at your own copy of the original source files.

A **self-contained working example** using simulated data is provided in `example/` and can be run without access to any of the original datasets.

---

## Quick Start (Working Example)

```julia
# Install dependencies (first time only)
include("example/simulate_example_data.jl")

# Run the full CVaR-OCS / MCMC-OCS pipeline on simulated data
include("example/run_example.jl")
```

This generates simulated genomic and phenotypic data resembling the Norway spruce dataset, runs CVaR-OCS and MCMC-OCS diagnostics, and produces example figures — all within approximately 5 minutes on a standard laptop.

---

## Citation

If you use this code, please cite:

```
Ahlinder, J. & Waldmann, P. (2026). Uncertainty-aware breeding decisions:
MCMC-based optimum contribution selection increases breeding decision robustness.
GENETICS. [DOI to be added upon publication]
```

---

## Contact

**Jon Ahlinder**
Department of Tree Breeding, Skogforsk
The Forestry Research Institute of Sweden
Uppsala, Sweden
jon.ahlinder@skogforsk.se

**Patrik Waldmann**
Mathematical Sciences, University of Oulu, Finland
