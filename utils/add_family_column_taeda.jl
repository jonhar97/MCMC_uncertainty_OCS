"""
Add Family Column to Loblolly Pine Phenotype Data
==================================================

Creates a Family column by combining Mom and Dad IDs.
Family is defined as the unique combination of female (Mom) and male (Dad) parents.

Author: Jon (Skogforsk)
Date: January 2026
"""

using CSV
using DataFrames
using Statistics

println("""
================================================================================
ADDING FAMILY COLUMN - LOBLOLLY PINE PHENOTYPES
================================================================================
""")

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData\\JWAS_phenotypes_926_index.txt"
OUTPUT_FILE = "C:\\Users\\JOAH\\OneDrive - Skogforsk\\Documents\\Projekt\\Optimum contribution selection\\TaedaPineData\\JWAS_phenotypes_926_index_with_family.txt"

println("Loading phenotype data...")

# ============================================================================
# LOAD DATA
# ============================================================================

df = CSV.read(INPUT_FILE, DataFrame)

println("✓ Data loaded")
println("  Total individuals: $(nrow(df))")
println("  Columns: $(names(df))")

# ============================================================================
# CREATE FAMILY COLUMN
# ============================================================================

println("\nCreating Family column from Mom and Dad...")

# Create Family as "Mom_x_Dad"
df.Family = string.(df.Mom) .* "_x_" .* string.(df.Dad)

println("✓ Family column created")

# ============================================================================
# SUMMARIZE FAMILY STRUCTURE
# ============================================================================

println("\nFamily structure summary:")

# Count unique families
n_families = length(unique(df.Family))
println("  Unique families: $n_families")

# Count unique moms and dads
n_moms = length(unique(df.Mom))
n_dads = length(unique(df.Dad))
println("  Unique moms: $n_moms")
println("  Unique dads: $n_dads")

# Family size distribution
family_sizes = combine(groupby(df, :Family), nrow => :family_size)
sort!(family_sizes, :family_size, rev=true)

println("\nFamily size statistics:")
println("  Mean family size: $(round(mean(family_sizes.family_size), digits=2))")
println("  Median family size: $(median(family_sizes.family_size))")
println("  Min family size: $(minimum(family_sizes.family_size))")
println("  Max family size: $(maximum(family_sizes.family_size))")

# Show top 10 largest families
println("\nTop 10 largest families:")
println("  Family                    Size")
println("  " * "-"^40)
for i in 1:min(10, nrow(family_sizes))
    println("  $(rpad(family_sizes.Family[i], 25)) $(family_sizes.family_size[i])")
end

# Family size distribution
println("\nFamily size distribution:")
size_dist = combine(groupby(family_sizes, :family_size), nrow => :n_families)
sort!(size_dist, :family_size)
for row in eachrow(size_dist)
    pct = (row.n_families / n_families) * 100
    println("  Size $(row.family_size): $(row.n_families) families ($(round(pct, digits=1))%)")
end

# ============================================================================
# REORDER COLUMNS (Family after ID, Mom, Dad)
# ============================================================================

println("\nReordering columns...")

# Get column order: ID, Mom, Dad, Family, then rest
other_cols = setdiff(names(df), ["ID", "Mom", "Dad", "Family"])
df = select(df, "ID", "Mom", "Dad", "Family", other_cols...)

println("✓ Columns reordered")
println("  New column order: $(join(names(df), ", "))")

# ============================================================================
# SAVE OUTPUT
# ============================================================================

println("\nSaving phenotype data with Family column...")
CSV.write(OUTPUT_FILE, df)

println("✓ Data saved: $OUTPUT_FILE")
println("  Total rows: $(nrow(df))")
println("  Total columns: $(ncol(df))")

# ============================================================================
# CREATE SUMMARY FOR USER
# ============================================================================

println("\n" * "="^80)
println("FAMILY COLUMN ADDITION COMPLETE!")
println("="^80)
println("\nSummary:")
println("  Individuals: $(nrow(df))")
println("  Families: $n_families")
println("  Moms: $n_moms")
println("  Dads: $n_dads")
println("  Mean family size: $(round(mean(family_sizes.family_size), digits=2))")
println("\nYou can now use this file for Figure 1:")
println("  PHENOTYPES_FILE = \"$OUTPUT_FILE\"")
println("  FAMILY_COLUMN = \"Family\"")
println("="^80)
