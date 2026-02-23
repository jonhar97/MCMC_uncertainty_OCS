# MCMC-Based Uncertainty-Aware Optimum Contribution Selection

[![Julia](https://img.shields.io/badge/Julia-1.9+-purple.svg)](https://julialang.org)
[![R](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code repository for the manuscript:

> **Uncertainty-aware breeding decisions: MCMC-based optimum contribution selection increases breeding decision robustness**  
> Jon Ahlinder & Patrik Waldmann  
> *Theoretical and Applied Genetics* (submitted 2026)

---

## Overview

Optimum Contribution Selection (OCS) is the standard framework for maximising genetic gain while constraining inbreeding in animal and plant breeding programs. Classical OCS relies on Maximum A Posteriori (MAP) point estimates of breeding values (EBVs) and treats them as known. This ignores the uncertainty inherent in Bayesian MCMC-based EBV estimation, which can be substantial — especially for individuals from small families or with sparse phenotyping.

This repository provides the full analysis pipeline demonstrating that **incorporating MCMC uncertainty into OCS decisions**:

- Improves individual selection robustness by **18–24%** with minimal genetic gain loss
- Enables principled risk stratification of candidates using robustness scores
- Reveals that MAP-OCS apparent within-family differentiation may reflect estimation error rather than true genetic differences

The approach is validated across two forest tree species:
- **Norway spruce** (*Picea abies*): n = 1,218 genotyped individuals across 126 families
- **Loblolly pine** (*Pinus taeda*): n = 926 individuals

---

## Repository Structure

```
MCMC_uncertainty_OCS/
│
├── README.md
│
├── preprocessing/               # Data preparation (run before src/)
│   ├── R/
│   │   ├── DataWrangling_spruce.Rmd      # Phenotype data setup, G/A/H matrix construction (spruce)
│   │   ├── Update_data_spruce.Rmd        # G matrix tuning, phenotype/matrix alignment (spruce)
│   │   └── Amatrix_taeda.Rmd             # A matrix construction for Loblolly pine
│   └── julia/
│       ├── JWAS_GBLUP_spruce_1218_G.jl   # JWAS MCMC-GBLUP, spruce 1218 genotypes (G matrix)
│       ├── JWAS_GBLUP_spruce_5022_H.jl   # JWAS MCMC-GBLUP, spruce 5022 individuals (H matrix)
│       ├── JWAS_GBLUP_spruce_5022_A.jl   # JWAS MCMC-GBLUP, spruce 5022 individuals (A matrix)
│       ├── JWAS_GBLUP_taeda_926_G.jl     # JWAS MCMC-GBLUP, pine 926 genotypes (G matrix)
│       └── JWAS_GBLUP_taeda_926_A.jl     # JWAS MCMC-GBLUP, pine 926 individuals (A matrix)
│
├── src/                         # Core MCMC-OCS analysis pipeline
│   ├── norway_spruce/
│   │   ├── MCMC_uncertainty_spruce.jl    # Main pipeline: MCMC-OCS, overlap analysis, GP regression
│   │   ├── mcmc_robustness_spruce.jl     # Robustness score calculation for all candidates
│   │   └── constrained_ocs_spruce.jl     # Constrained OCS excluding high-risk individuals
│   └── loblolly_pine/
│       ├── MCMC_uncertainty_taeda.jl     # Main pipeline: MCMC-OCS, overlap analysis, GP regression
│       ├── calculate_robustness_taeda.jl # Robustness score calculation for top 200 candidates
│       └── constrained_ocs_taeda.jl      # Constrained OCS excluding high-risk individuals
│
├── figures/                     # Publication figure generation
│   ├── norway_spruce/
│   │   ├── Figure1_within_family_uncertainty.jl  # Posterior KDE plots by family (Fig. 1)
│   │   ├── Figure3_risk_assessment.jl             # Risk stratification of MAP-OCS selections (Fig. 3)
│   │   ├── Figure4_dual_metrics.jl                # Dual metrics comparison (Fig. 4)
│   │   └── JWAS_diagnostic_plots_spruce.jl        # MCMC convergence and EBV diagnostics
│   └── loblolly_pine/
│       ├── Figure1_within_family_uncertainty.jl   # Posterior KDE plots by family (Fig. 1)
│       ├── Figure3_risk_assessment.jl              # Risk stratification of MAP-OCS selections (Fig. 3)
│       └── Figure4_dual_metrics.jl                 # Dual metrics comparison (Fig. 4)
│
├── utils/                       # Helper scripts and secondary analyses
│   ├── add_family_column_taeda.jl         # Derive family IDs from parent information (pine)
│   ├── add_quartiles_taeda.jl             # Assign robustness quartiles (pine)
│   ├── identify_families_taeda.jl         # Select representative families for Fig. 1 (pine)
│   ├── overlap_analysis_GP.jl             # GP regression on EBV vs selection frequency
│   ├── analyze_norway_spruce_ebv_selection.jl  # Polynomial/GP regression, EBV-frequency relationship
│   └── load_and_analyze_norway_spruce.jl  # Load MCMC contributions and run regression analysis
│
└── example/                     # Self-contained working example (simulated data)
    ├── README.md
    ├── simulate_example_data.jl
    └── run_example.jl
```

---

## Analysis Workflow

The analysis proceeds in four stages. Run them in order:

```
Stage 1: preprocessing/R/       →  Construct A, G, H matrices; wrangle phenotypes
Stage 2: preprocessing/julia/   →  Fit MCMC-GBLUP in JWAS; save posterior EBV samples
Stage 3: src/                   →  Run MCMC-OCS, calculate robustness scores, constrained OCS
Stage 4: figures/               →  Generate publication figures from saved results
```

---

## Key Methods

### MCMC-OCS

For each MCMC iteration $t$, we solve the OCS problem:

$$\max_{\mathbf{c}^{(t)}} \; \mathbf{c}^{(t)\top} \hat{\mathbf{g}}^{(t)} \quad \text{subject to} \quad \frac{1}{2}\mathbf{c}^{(t)\top} \mathbf{G} \, \mathbf{c}^{(t)} \leq \theta, \quad \mathbf{1}^\top \mathbf{c}^{(t)} = 1, \quad \mathbf{c}^{(t)} \geq \mathbf{0}$$

where $\hat{\mathbf{g}}^{(t)}$ is the vector of EBVs drawn from the posterior at iteration $t$, $\mathbf{G}$ is the genomic relationship matrix, and $\theta$ is the coancestry constraint.

This produces a distribution of OCS solutions across iterations rather than a single point solution, from which individual robustness scores and population-level stability metrics are derived.

### Robustness Scores

Individual robustness is quantified as the genetic gain loss when an individual is excluded from the candidate pool:

$$r_i = \frac{\Delta G_{-i}}{\Delta G_{\text{full}}}$$

where $\Delta G_{-i}$ is the gain achievable when individual $i$ is excluded. Individuals with $r_i$ close to 1 are robust (easily replaceable); those with $r_i \gg 1$ represent high-risk dependencies.

### Constrained OCS

High-risk individuals (bottom 25% of MAP-OCS selected by robustness score) are excluded, and OCS is re-run on the remaining candidates. This produces a more robust selection with typically < 2% genetic gain loss.

---

## Dependencies

### Julia packages

```julia
using Pkg
Pkg.add([
    "JWAS",          # Bayesian mixed model MCMC
    "JuMP",          # Mathematical programming
    "COSMO",         # Quadratic programming solver
    "DataFrames",
    "CSV",
    "Statistics",
    "LinearAlgebra",
    "Plots",
    "StatsPlots",
    "KernelDensity",
    "AbstractGPs",   # Gaussian process regression
    "KernelFunctions",
    "Optim",
    "HypothesisTests",
    "Distributions",
    "DelimitedFiles",
    "JLD2",
    "FileIO",
    "ProgressMeter",
    "Printf"
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

The Norway spruce phenotypic and pedigree data are the property of Skogforsk (The Forestry Research Institute of Sweden) and cannot be publicly released. The Loblolly pine dataset is derived from the publicly available NCBI SRA data described in the manuscript.

A **self-contained working example** using simulated data is provided in `example/` and can be run without access to the original datasets.

---

## Quick Start (Working Example)

```julia
# Install dependencies (first time only)
include("example/simulate_example_data.jl")

# Run the full MCMC-OCS uncertainty pipeline on simulated data
include("example/run_example.jl")
```

This will generate simulated genomic and phenotypic data resembling the Norway spruce dataset, run MCMC-OCS across 500 iterations, calculate robustness scores, and produce example figures — all within approximately 5 minutes on a standard laptop.

---

## Citation

If you use this code, please cite:

```
Ahlinder, J. & Waldmann, P. (2026). Uncertainty-aware breeding decisions:
MCMC-based optimum contribution selection increases breeding decision robustness.
Theoretical and Applied Genetics. [DOI to be added upon publication]
```

---

## Contact

**Jon Ahlinder**  
Department of Tree Breeding, Skogforsk  
The Forestry Research Institute of Sweden  
Uppsala, Sweden

**Patrik Waldmann**  
University of Oulu, Finland
