# Attention Kernel Contract for Wavefront-64 GPUs

This document defines the **non-negotiable architectural constraints** that any high-performance attention kernel must satisfy on **AMD wavefront-64 GPUs**.

This is a **kernel contract**, not an optimization guide.

If a kernel violates these constraints, performance degradation is not accidental — it is architecturally guaranteed.

---

## 1. Scope and Intent

This contract applies to:

- FlashAttention-style kernels
- Fused attention implementations
- Any kernel combining:
  - Matrix multiplication
  - Softmax
  - Reduction and normalization
  - Persistent accumulators

The contract is defined **at the ISA and execution-model level**, independent of:
- Triton
- HIP
- ROCm compiler versions

---

## 2. Wavefront-64 Execution Semantics

AMD GPUs execute instructions in **64-thread wavefronts**.

This has three immediate consequences:

1. All 64 lanes issue instructions in lockstep
2. Register usage scales with wave width
3. Reduction and cross-lane operations have higher fixed cost than warp-32 designs

### Contract Requirement 2.1 — Full-Wave Residency

Any kernel expecting to hide memory or pipeline latency **must sustain ≥2 resident wavefronts per SIMD**.

Failure to meet this requirement results in:
- MFMA pipeline bubbles
- Exposed global memory latency
- Reduced throughput even in compute-bound kernels

---

## 3. Register Residency Contract

### 3.1 VGPR and AccVGPR Separation

AMD CDNA architectures physically separate register files:

| Register File | Purpose |
|--------------|--------|
| ArchVGPRs | Addressing, loads, scalar/vector ALU, softmax |
| AccVGPRs | MFMA accumulators only |

This separation is enforced by hardware.

AccVGPRs:
- Cannot participate in general ALU operations
- Must be explicitly read via `v_accvgpr_read`
- Consume register budget per wavefront

---

### Contract Requirement 3.1 — Accumulator Residency Budget

For a kernel to sustain ≥2 wavefronts per SIMD:

- **Total live ArchVGPRs per lane must remain below occupancy thresholds**
- **Total live AccVGPRs must not monopolize the MFMA register file**

Long-lived accumulators are therefore a **first-class occupancy constraint**, not an implementation detail.

---

## 4. MFMA Usage Contract

### 4.1 MFMA Operand Rules

MFMA instructions impose strict operand placement:

- A, B operands → ArchVGPRs
- Accumulator (C) → AccVGPRs
- Result (D) → AccVGPRs

Matrix results are **trapped in AccVGPRs** until explicitly moved.

---

### Contract Requirement 4.1 — Domain Crossing Is Explicit

Any kernel performing:
- Exponentiation
- Reduction
- Normalization
- Masking

**must transfer data from AccVGPRs to ArchVGPRs**.

This transfer:
- Has non-zero latency
- Competes with MFMA scheduling
- Cannot be optimized away by the compiler

---

## 5. Softmax Dependency Contract

### 5.1 Online Softmax Implications

Online softmax requires:

- Persistent running max
- Persistent normalization denominator
- Re-scaling of the output accumulator

On wavefront-64, this implies:

- Long-lived ArchVGPR state
- Repeated AccVGPR → ArchVGPR transfers
- Extended register live ranges

---

### Contract Requirement 5.1 — Softmax State Lifetime

Any kernel using online softmax **must account for**:

- Softmax state remaining live across all KV tiles
- Accumulator rescaling preventing early writeback
- Register pressure compounding across wavefront width

If these costs are ignored, occupancy collapse is guaranteed.

---

## 6. Reduction and Synchronization Contract

### 6.1 Reduction Cost Scaling

In wavefront-64:

- Reductions require more steps than warp-32
- LDS usage is amplified
- Bank conflicts scale with wave width

---

### Contract Requirement 6.1 — Reduction Awareness

Attention kernels **must not assume** warp-32 reduction costs.

Designs relying on frequent cross-lane reductions:
- Pay higher fixed overhead on AMD
- Become latency-sensitive under register pressure

---

## 7. Failure Modes (Non-Optional)

If the above contracts are violated, the following failure modes occur:

- **Occupancy collapse** (1 wave/SIMD or less)
- **MFMA starvation** due to accumulator reads
- **Inability to hide global memory latency**
- **LDS congestion during reductions**
- **Performance cliffs instead of gradual degradation**

These outcomes are **architectural**, not compiler bugs.

---

## 8. Design Implications

An attention kernel that respects wavefront-64 must:

- Treat accumulator lifetime as a constrained resource
- Minimize persistent ArchVGPR and AccVGPR state
- Accept increased bandwidth to preserve occupancy
- Explicitly pipeline MFMA and accumulator reads
- Avoid assuming NVIDIA warp-32 behavior

---

## 9. Contract Summary (Non-Negotiable)

| Constraint | Required |
|----------|---------|
| ≥2 waves per SIMD | Yes |
| Explicit AccVGPR ↔ ArchVGPR movement | Yes |
| Bounded accumulator lifetime | Yes |
| Reduction cost awareness | Yes |
| Occupancy-first design | Yes |

---

## 10. Closing Statement

FlashAttention’s design is **algorithmically sound**, but its performance is contingent on architectural assumptions.

On wavefront-64 GPUs, those assumptions become **contracts**.

Any attention kernel that ignores this contract will fail — predictably, repeatedly, and non-accidentally.

---

## Status

This document defines **constraints only**.

Subsequent documents will:
- Identify when FlashAttention violates this contract
- Explore alternative designs that satisfy it
- Quantify tradeoffs between bandwidth, occupancy, and correctness
