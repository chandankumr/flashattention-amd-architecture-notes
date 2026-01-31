# FlashAttention AMD Architecture Notes

This repository contains low-level architectural and performance analysis of FlashAttention kernels, with a focus on AMD GPU execution characteristics (VGPR pressure, wavefront occupancy, and compiler behavior).

## Contents

- [Register Pressure vs Occupancy](docs/flashattention_registers_vs_occupancy.md)
- [Quantifying VGPR Usage in FlashAttention](docs/flashattention_vgpr_analysis.md)
- [Extracting Real VGPR Counts from Triton AMD Kernels](docs/extracting_real_vgpr_counts.md)

## Scope

- No benchmarks without architectural explanation
- Focus on kernel structure, register lifetimes, and execution models
- Comparisons only when they reveal structural tradeoffs
