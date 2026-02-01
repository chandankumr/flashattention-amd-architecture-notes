# ISA-Level Attention Mapping: MFMA, AGPR, and the Accumulator Contract on AMD GPUs

This document analyzes FlashAttention-style kernels from the **instruction set (ISA)** and **register-file** perspective on AMD CDNA architectures.

The goal is to expose **physical hardware constraints** that shape kernel behavior, not to discuss algorithmic optimizations or benchmarking results.

---

## 1. Architectural Context: Why ISA-Level Analysis Matters

Most attention kernel discussions stop at:

- Tensor shapes
- Memory bandwidth
- High-level tiling strategies

However, on AMD GPUs, **performance is frequently determined by ISA-level register residency and instruction scheduling**, especially for matrix-heavy workloads.

FlashAttention is a useful case study because it:

- Uses large, persistent accumulators
- Mixes matrix math with scalar/vector math (softmax)
- Places sustained pressure on register files and occupancy

Understanding its behavior requires looking **below Triton and LLVM**, at the **MFMA execution model** and **register-file partitioning** enforced by hardware.

---

## 2. Physical Divergence: NVIDIA vs AMD Register Models

### 2.1 NVIDIA (Warp-32 Model)

NVIDIA GPUs operate with a unified register model:

- Single unified register file
- Tensor Core instructions read/write general registers
- Accumulators are regular registers
- Compiler manages pressure mainly via spilling

This model tolerates **long-lived accumulators** relatively well.

---

### 2.2 AMD CDNA (Wavefront-64 Model)

AMD CDNA architectures (e.g., MI200 / MI300) differ fundamentally.

The register file is **physically split**:

| Register Type | Purpose |
|--------------|--------|
| **ArchVGPRs** | General vector registers (addressing, arithmetic, loads, softmax) |
| **AccVGPRs** | Dedicated accumulator registers for MFMA matrix operations |

This split is **enforced by hardware**, not a compiler abstraction.

> **AccVGPRs cannot directly participate in general ALU operations.**

This constraint is central to understanding attention kernel behavior on AMD GPUs.

---

## 3. MFMA: The Primitive of Matrix Compute on AMD

The fundamental matrix instruction is MFMA:

v_mfma_f32_16x16x16f16 D, A, B, C


### Operand placement rules:

- **A, B** → ArchVGPRs
- **C (accumulator)** → AccVGPRs
- **D (result)** → AccVGPRs

Implications:

- Matrix results are **trapped in AccVGPRs**
- Any non-matrix operation requires explicit movement
- Accumulator residency becomes a first-class performance constraint

---

## 4. Domain-Crossing Cost: AccVGPR ↔ ArchVGPR

FlashAttention performs the following sequence:

Q × Kᵀ → Softmax → Multiply by V


### The problem


To apply softmax, data must be moved:

```asm
v_accvgpr_read_b32 v10, a0
v_accvgpr_read_b32 v11, a1
v_exp_f32 v10, v10
```

> Softmax operations (exp, sum, normalization) **cannot operate on AccVGPRs**.

This is because AccVGPRs are only addressable by MFMA-class instructions and are not visible to the general vector ALU pipeline.

This introduces:

- Explicit data movement

- Instruction latency

- Scheduling pressure

Unlike CUDA’s unified register model, this cost is architecturally unavoidable on AMD CDNA due to the physical separation between AccVGPRs and ArchVGPRs.


---

### Execution Flow: MFMA and Softmax Across Register Domains

1. **ArchVGPRs provide matrix operands**
   - Query (`Q`) and Key (`K`) tiles are loaded into **ArchVGPRs**
   - These registers feed the MFMA pipeline

2. **MFMA executes matrix multiplication**
   - `v_mfma_*` instructions consume operands from ArchVGPRs
   - Accumulation occurs exclusively in **AccVGPRs**

3. **Results are stored in AccVGPRs**
   - Intermediate attention scores reside in AccVGPRs
   - They cannot be directly used by general ALU instructions

4. **Domain crossing via `v_accvgpr_read`**
   - Accumulator values are explicitly transferred from AccVGPRs to ArchVGPRs
   - This enables non-matrix operations

5. **Softmax executes in ArchVGPRs**
   - Exponentiation, reduction, and normalization occur using ALU instructions
   - Results may later be re-used as MFMA inputs or written to memory



### Register-Domain Flow

ArchVGPRs  
→ provide A, B operands  
→ MFMA executes  
→ results stored in AccVGPRs  
→ `v_accvgpr_read` transfers data  
→ Softmax executes in ArchVGPRs  
       
---

## 5. Accumulator Residency and Pipeline Hazards

Because MFMA accumulators live in AccVGPRs:

- Keeping data in AccVGPRs is beneficial for matrix math

- Softmax forces repeated domain crossings

- Naive scheduling can stall MFMA units during data movement

High-performance kernels must **pipeline these operations:**

```css
Time →
Tile n:     MFMA(Q,K) → AccVGPR
Tile n-1:   AccVGPR → ArchVGPR → Softmax
```
This pipelining requires sufficient wave occupancy; otherwise, MFMA units may idle while waiting on accumulator reads and softmax execution.

This requires:

- Careful instruction scheduling
- Sufficient wave occupancy to hide latency

---

## 6. Wavefront-64 Amplification Effects

AMD executes **64 threads per wavefront.**

Consequences:

- Register usage is multiplied across 64 lanes
- Occupancy drops sharply once register limits are exceeded
- Latency hiding is more sensitive to VGPR and AccVGPR pressure

For attention kernels, this means:

- Large accumulators scale poorly
- Long-lived state is disproportionately expensive

This is a **structural effect**, not a tuning artifact.

---

## 7. FlashAttention’s Accumulator Footprint at ISA Level

Typical FlashAttention configurations use:

- FP32 accumulators
- Large tile sizes (e.g., 128 × 64)

Effects:

- Accumulators occupy a large fraction of AccVGPRs
- Softmax state occupies ArchVGPRs
- Pointer arithmetic and masking consume additional VGPRs

Resulting behavior:

- Limited wave residency
- Reduced ability to hide memory and instruction latency

This outcome is predictable **from ISA constraints alone.**

---

## 8. The Core Architectural Tension

FlashAttention optimizes for:

- Minimal memory traffic
- High data reuse
- Long-lived accumulators

AMD CDNA hardware prefers:

- Shorter live ranges
- Higher wave occupancy
- Explicit movement between compute domains

These goals are **in fundamental tension.**

---

## 9. Implications for Attention Kernel Design

From an ISA perspective, AMD-friendly attention kernels should:

- Minimize persistent AccVGPR residency
- Limit long-lived ArchVGPR state
- Treat accumulator lifetime as a first-class constraint
- Accept bandwidth increases to preserve occupancy
- Explicitly schedule MFMA and AccVGPR reads

This naturally motivates:

- Partial accumulator writeback (Split-K)
- Two-phase softmax
- Architecture-aware kernel contracts

---

## 10. Key Insight

The performance challenges of FlashAttention on AMD GPUs are not primarily caused by:

- Triton
- ROCm
- Compiler immaturity

They arise from a **mismatch between FlashAttention’s accumulator-heavy design and AMD’s split register-file execution model.**

Recognizing this mismatch is the foundation for any meaningful redesign.

---

## 11. Why This Matters

Understanding MFMA, AccVGPRs, and register residency is essential for:

- Designing AMD-first attention kernels
- Writing effective kernel contracts
- Interpreting compiler output correctly
- Making informed bandwidth vs occupancy tradeoffs

This document establishes the **physical constraints** that later design decisions must respect.

---

## Status

This document intentionally focuses on **hardware mechanics**, not solutions.

Subsequent documents will build on this foundation to:

- Define wavefront-64 kernel contracts
- Evaluate compiler behavior using RGA
- Propose accumulator-lifetime-aware designs