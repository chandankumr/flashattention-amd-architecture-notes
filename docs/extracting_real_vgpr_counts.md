# Extracting Real VGPR Counts from Triton AMD Kernels

This document outlines a practical workflow for extracting **actual VGPR usage** from Triton-generated AMD GPU kernels, rather than relying on high-level estimates or compiler heuristics.

The goal is to connect:
- Triton kernel structure
- LLVM code generation
- Final ISA-level VGPR allocation

---

## Why VGPR Counts Must Be Measured Post-Compilation

In Triton, register usage is not explicitly declared. VGPR consumption emerges from:
- Tensor shapes and lifetimes
- Loop structure
- Compiler register allocation decisions
- Wavefront width (64 threads on AMD)

High-level reasoning is useful, but **occupancy decisions are made using final ISA VGPR counts**, not Triton IR.

---

## Overview of the Compilation Pipeline

Triton Python
↓
Triton IR (TTIR)
↓
LLVM IR (amdgpu)
↓
GCN / RDNA ISA
↓
VGPR allocation & occupancy


VGPR usage can only be trusted **after** the LLVM → ISA stage.

---

## Step 1: Force Triton to Emit Compiled Artifacts

Set the following environment variables before running your Triton kernel:

```bash
export TRITON_CACHE_DIR=/tmp/triton_cache
export TRITON_DUMP_ASM=1
export TRITON_DUMP_LLVM=1
```

Run your FlashAttention workload once. Triton will populate the cache with:

- LLVM IR
- Compiled HSACO objects
- Metadata

## Step 2: Locate the Compiled HSACO Binary

Inside the cache directory:

```bash
find /tmp/triton_cache -name "*.hsaco"
```

Each `.hsaco` corresponds to a compiled kernel specialization (tile sizes, dtypes, etc.).

Identify the kernel of interest by timestamp or hash.

---

## Step 3: Disassemble the Kernel

Use AMD’s disassembler:

```bash
llvm-objdump --arch=amdgcn --mcpu=gfx90a -d kernel.hsaco > kernel.s
```

(Replace `gfx90a` with your target architecture if needed.)

---

## Step 4: Extract VGPR Usage

Search for the kernel metadata block in the disassembly:

```bash
grep -A20 ".amdhsa_kernel" kernel.s
```

Look for fields such as:

```
.sgpr_count: 64
.vgpr_count: 192
.wavefront_size: 64
```

👉 `vgpr_count` **is the number that directly constrains occupancy.**

---

## Step 5: Compute Theoretical Occupancy

On AMD GPUs, occupancy is limited by:

- VGPRs per CU
- Wavefront size (64)
- Max wavefronts per CU

Example (CDNA-like):

```sql
Total VGPRs per CU: 65536
VGPRs per wavefront = vgpr_count × 64
Max wavefronts = floor(65536 / (vgpr_count × 64))
```

Example:

```sql
vgpr_count = 192
Per-wavefront = 192 × 64 = 12288
Max wavefronts ≈ 5
```

This directly explains observed occupancy cliffs.

---

## Step 6: Correlate VGPR Usage with Kernel Structure

Once VGPR usage is known, map it back to Triton constructs:

| Source                        | Effect on VGPRs  |
| ----------------------------- | ---------------- |
| Persistent `acc` tensor       | Large base cost  |
| Online softmax (`m_i`, `l_i`) | Long-lived rows  |
| Large BLOCK_M / BLOCK_DMODEL  | Quadratic growth |
| Loop-carried accumulators     | Prevent reuse    |
| Masked paths                  | Extra live state |

This correlation is the real insight — not the number itself.

---

## Common Pitfalls

- **Nsight Compute ≠ VGPR truth** on AMD
- Triton autotune may hide high-VGPR variants
- Debug builds inflate register usage
- Multiple kernel variants exist — inspect the right one

---

## Takeaway

VGPR pressure in Triton FlashAttention kernels is not hypothetical.
It is measurable, reproducible, and directly tied to kernel structure.

Extracting ISA-level VGPR counts is the only reliable way to reason about occupancy and performance on AMD GPUs.
