# Quantifying VGPR Pressure in FlashAttention (Triton) on Wavefront-64 GPUs

## Overview

FlashAttention achieves high performance by trading global memory bandwidth for on-chip state, using an **online softmax** formulation with **incremental accumulation** of the output. While this design maps well to NVIDIA warp-32 architectures, it can introduce severe **register pressure** on AMD GPUs, which execute **wavefronts of 64 threads**.

This document quantifies the VGPR usage of a representative FlashAttention forward kernel implemented in Triton (`flash_attn_triton_amd/fwd_prefill.py`) and explains why occupancy collapses more aggressively on AMD hardware.

The goal is not to criticize the design, but to make the architectural tradeoffs explicit.

---

## Kernel Configuration Analyzed

We analyze a commonly autotuned configuration used in practice:

- `BLOCK_M = 128`
- `BLOCK_N = 128`
- `BLOCK_DMODEL = 128`
- `num_warps = 8` (AMD: 8 wavefronts → 512 threads per workgroup)
- Accumulation type: `fp32`

This configuration maximizes arithmetic intensity but also maximizes long-lived register state.

---

## Long-Lived State in the Kernel

The following tensors are **persistent across the entire KV loop**, creating long register lifetimes.

### 1. Output Accumulator (`acc`)

```python
acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=fp32)


----------------------------------------
    Shape: 128 × 128

    Total elements: 16,384 fp32

    Distributed across 512 threads:
-------------------------------------------


16,384 / 512 = 32 fp32 values per thread

→ ~32 VGPRs per thread

This accumulator is:
    initialized once
    rescaled and updated in every KV iteration
    never written back or cleared until the epilogue

----------------------------------------------------------------------------------------

2. Online Softmax State (m_i, l_i)
    m_i = tl.full([BLOCK_M], -inf, fp32)
    l_i = tl.full([BLOCK_M], 1.0, fp32)

    Total elements: 2 × 128 = 256 fp32
    Per-thread share:
        256 / 512 = 0.5 fp32 → rounded up

    → ~1 VGPR per thread

    These values must remain live to correctly rescale the accumulator when a new maximum is encountered.

----------------------------------------------------------------------------------------

3. Query Fragment (q)
    q = tl.load(q_ptrs)

    Shape: 128 × 128
    Reused in every tl.dot(q, k) call
    Triton typically promotes to fp32 fragments for dot products

    16,384 / 512 = 32 fp32 values per thread


    → ~32 VGPRs per thread

----------------------------------------------------------------------------------------

Peak Register Pressure Contributors

4. Score Accumulator (acc_s)
    acc_s = tl.dot(q, k)

    Shape: 128 × 128
    Temporary, but overlaps in lifetime with acc, m_i, and q

    Contribution:
        16,384 / 512 = 32 fp32 values per thread

    → ~32 VGPRs per thread (peak)

Estimated VGPR Usage Per Thread
| Component                   | VGPRs              |
| --------------------------- | ------------------ |
| Output accumulator (`acc`)  | ~32                |
| Query fragment (`q`)        | ~32                |
| Softmax state (`m_i,l_i`)   | ~1                 |
| Score accumulator (`acc_s`) | ~32                |
| Pointers & control          | ~10–15             |
| **Total (conservative)**    | **~110–120 VGPRs** |

----------------------------------------------------------------------------------------

Mapping to AMD CDNA Hardware

Assumptions:
    VGPRs per CU ≈ 65,536
    Wavefront size = 64 threads

VGPRs per wavefront:
    120 VGPRs/thread × 64 threads = 7,680 VGPRs

Maximum resident wavefronts per CU:
    65,536 / 7,680 ≈ 8.5 → at most 8 wavefronts


After accounting for:
    LDS usage
    SGPR pressure
    instruction constraints

Practical occupancy often drops to ~4–6 wavefronts per CU.

Why This Hurts AMD More Than NVIDIA

On NVIDIA GPUs:
    Warp size = 32 threads
    Same per-thread register usage

120 × 32 = 3,840 registers per warp


This is half the per-scheduling-unit pressure, allowing more warps to remain resident and improving latency hiding.

The algorithm is the same; the architectural scaling is not.

Implications

    The register pressure is algorithmic, not accidental

    Triton preserves the same online-softmax accumulator lifetime as the CUDA kernel

    Wavefront-64 architectures amplify long-lived register state

    Performance portability requires architecture-aware restructuring, not micro-optimizations

Potential Mitigation Directions (Not Solutions)

    Smaller BLOCK_M for wavefront-64 GPUs

    Split-KV or partial-accumulator designs

    Two-pass softmax variants for long sequences

    Wave32-specialized kernels where supported

Each trades arithmetic intensity for improved occupancy.

Conclusion

    FlashAttention’s design deliberately prioritizes bandwidth efficiency by keeping large accumulators and softmax state on-chip. On AMD GPUs, the combination of wavefront-64 execution and large persistent accumulators leads to sharp occupancy cliffs. Quantifying this effect helps guide future ROCm-optimized kernel designs.