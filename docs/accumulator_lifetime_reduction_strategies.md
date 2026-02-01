# Accumulator Lifetime Reduction Strategies for AMD GPUs

This document explores design alternatives to reduce long-lived accumulator register pressure in FlashAttention-style kernels on wavefront-64 architectures, without breaking correctness.

AMD GPUs are more sensitive to prolonged live registers because a single wavefront carries 64 threads. High persistent register usage can sharply reduce occupancy and degrade latency hiding. This document evaluates strategies that trade memory bandwidth and compute cost for reduced register live ranges.

---

## 1. Problem Summary

Refer to the case study [(`flashattention_vgpr_case_study_128x64.md`)](flashattention_vgpr_case_study_128x64.md) for a detailed analysis of how a large persistent output accumulator and associated online softmax state generate high VGPR usage.

The core challenge is:
- The accumulator (`acc`) and softmax state (`m_i`, `l_i`) are **live across all KV blocks**
- This extended live range forces many registers to remain allocated simultaneously
- On AMD, this directly reduces the number of resident wavefronts

---

## 2. Design Strategy Overview

We explore the following directions:

### A. Partial Accumulator Writeback (Split-K Strategy)
Instead of holding the full accumulator in registers across the KV loop:
- Break the sequence into smaller chunks
- Accumulate block results into shared memory or LDS
- Write partial results to global memory periodically
- Reload and merge them as needed
- Pros → reduced persistent register pressure  
- Cons → extra memory traffic

### B. Two-Phase Softmax
Separate softmax into:
- Phase 1 — compute per-tile contributions (max + sum)
- Phase 2 — final normalization
This avoids keeping softmax state live across all KV blocks at once.

### C. Re-materialization of Query Fragments
Instead of keeping the whole query tile live across all KV tiles, reload or recompute slices as needed:
- Lower the live range
- Reduce register pressure
- Potential tradeoff: more loads

### D. Smaller Tile Sizes
By reducing tile sizes (e.g., `BLOCK_M=64`), we:
- Reduce register footprint
- Increase number of tiles
- Potential performance impact on arithmetic intensity

---

## 3. Strategy Tradeoffs

| Strategy | Reduces VGPRs? | Extra Bandwidth? | Complex to Implement |
|----------|----------------|----------------|---------------------|
| Split-K | Moderate | Yes | Medium |
| Two-Phase Softmax | High | Yes | High |
| Re-materialization | Moderate | Yes | Low |
| Smaller Tiles | Yes | Yes | Low |

Discussion:
- Strategies that reduce live ranges often increase memory bandwidth
- The goal is to find a **sweet spot** where the occupancy improvement outweighs the bandwidth cost

---

## 4. Proposed Implementation Concepts

### 4.1 Partial Writeback Pseudocode
acc_partial = zero
for blocks in [0:x):
acc_partial += dot(q, v)
if blocks % split_interval == 0:
write_to_global(acc_partial)
acc_partial = load_global_partial()


Explanation:
- Perform partial accumulation
- Write intermediate results to global memory
- Reload and merge when needed

---

## 5. Evaluation Plan

To assess these strategies, for each we should:

1. Generate Triton variants with modified accumulator lifetimes
2. Compile and extract VGPR counts
3. Measure occupancy
4. Benchmark performance with representative model dimensions
5. Compare against baseline

---

## 6. Design Strategy Summary

The following table summarizes the primary design strategies explored to mitigate long-lived register pressure in FlashAttention-style kernels on AMD GPUs.

| Strategy                    | Core Idea                                | VGPR Reduction | Occupancy Impact | Bandwidth Cost | Implementation Complexity |
| --------------------------- | ---------------------------------------- | -------------- | ---------------- | -------------- | ------------------------- |
| Baseline (Online Softmax)   | Persistent accumulator + running softmax | None           | Poor on AMD      | Low            | Existing                  |
| Smaller Tiles               | Reduce BLOCK_M / BLOCK_DMODEL            | Moderate       | Moderate         | Higher         | Low                       |
| Re-materialize Q            | Reload Q instead of keeping live         | Moderate       | Moderate         | Higher         | Low                       |
| Split-K (Partial Writeback) | Periodic accumulator spill               | Moderate–High  | High             | Higher         | Medium                    |
| Two-Phase Softmax           | Separate score + value passes            | High           | High             | High           | High                      |
| Split-K + Two-Phase         | Combine both techniques                  | Very High      | Very High        | Very High      | Very High                 |


This table provides a high-level comparison. Detailed analysis of each strategy follows in subsequent sections.

---

## 7. Partial Accumulator Writeback (Split-K) — Deep Dive
### 7.1 Motivation

The dominant contributor to VGPR pressure in FlashAttention forward kernels is the **long-lived output accumulator:**

- Shape: `(BLOCK_M × HEAD_DIM)`
- Lifetime: entire KV traversal
- Data type: fp32
- Residency: registers

On wavefront-64 architectures, this accumulator alone can consume hundreds of VGPRs per lane, sharply limiting occupancy.

**Key observation:**
The accumulator does not need to remain fully live across the entire KV loop if we are willing to:

- Materialize partial results
- Re-accumulate them later in a numerically correct way

This motivates a **Split-K / Partial Writeback** strategy.

---

7.2 Conceptual Idea

Instead of accumulating over the full K dimension in one pass:
```ini
acc = Σ softmax(QKᵀ) · V   over K = [0 … K_total)
```

We split the K dimension into **K partitions:**

```mathematica
K = K₀ ∪ K₁ ∪ … ∪ Kₙ
```

And compute:

```ini
acc_i = Σ softmax_i(QKᵀ) · V   over K_i
```

Each `acc_i` is:

- Smaller in lifetime
- Can be written out early
- Recombined later using correct softmax rescaling

### 7.3 Why This Reduces Register Pressure
#### Baseline (current FlashAttention)

- acc lives across all KV blocks
- m_i, l_i must also persist
- Registers cannot be freed or reused
- VGPR pressure scales with BLOCK_M × HEAD_DIM

#### Split-K Variant

- acc_partial lives only across a subset of KV blocks

- After writeback:
    - Registers are released
    - New accumulation begins

- Softmax state is localized per split
- Peak live register set is significantly reduced

👉 This trades **register lifetime** for **extra memory traffic.**

On AMD GPUs, this is often a favorable trade.

---

### 7.4 Correctness: How Softmax Still Works

The main challenge is **softmax correctness.**

Recall online softmax identity:

For blocks A and B:

```ini
m = max(m_A, m_B)
l = exp(m_A - m) * l_A + exp(m_B - m) * l_B
acc = exp(m_A - m) * acc_A + exp(m_B - m) * acc_B
```

#### Split-K preserves correctness by storing:

For each split:

- `acc_i` (partial output accumulator)
- `m_i` (local max)
- `l_i` (local normalization)

Later, we combine splits using the identity above.

**➡️ Numerically identical to full attention**, up to floating-point roundoff.

---

### 7.5 Concrete Kernel Structure (Conceptual)
#### Phase 1: Partial accumulation

```
for each K-split:
    acc_partial = 0
    m_partial = -inf
    l_partial = 0

    for K-blocks in this split:
        compute QKᵀ
        update m_partial, l_partial
        acc_partial += P · V

    write acc_partial, m_partial, l_partial to memory
```

#### Phase 2: Reduction / merge

```
acc = 0
m = -inf
l = 0

for each split:
    rescale existing acc using (m, l)
    rescale split acc using (m_i, l_i)
    merge accumulators
```

This merge can be:

- A second kernel
- A fused epilogue
- Or done by fewer wavefronts

---

### 7.6 Triton-Specific Mapping

In Triton terms:

**What changes**

- `acc` becomes `acc_partial`
- KV loop range is shortened
- Softmax state (`m_i`, `l_i`) scoped per split
- Intermediate results written to global memory

**What stays the same**

- Dot product structure
- Masking
- Numerical stability guarantees

Pseudocode (Triton-style)

```python
for split_id in range(num_splits):
    acc = tl.zeros([BLOCK_M, D], fp32)
    m_i = tl.full([BLOCK_M], -inf)
    l_i = tl.zeros([BLOCK_M])

    for start_n in split_range(split_id):
        qk = tl.dot(q, k)
        m_new = max(m_i, max(qk))
        l_i = l_i * exp(m_i - m_new) + sum(exp(qk - m_new))
        acc = acc * exp(m_i - m_new) + tl.dot(p, v)
        m_i = m_new

    store(acc, m_i, l_i)
```

---

### 7.7 Expected Impact on VGPR Usage

| Component              | Baseline  | Split-K    |
| ---------------------- | --------- | ---------- |
| Accumulator live range | Full KV   | Partial KV |
| Peak VGPRs             | Very high | Lower      |
| Occupancy              | Low       | Higher     |
| Bandwidth              | Lower     | Higher     |
| Control complexity     | Low       | Medium     |

The **key win** is that peak live registers drop, allowing:

- More resident wavefronts
- Better latency hiding
- Improved throughput on AMD GPUs

---

### 7.8 When Split-K Makes Sense (and When It Doesn’t)
#### Good fit

- Long sequences
- Large head dimensions
- Occupancy-limited kernels
- CDNA / RDNA GPUs

#### Poor fit

- Very short sequences
- Memory-bandwidth-limited regimes
- Extremely small head dims

This strategy should be **selectively applied**, not universally.

### 7.9 Relationship to Existing Work

Split-K techniques are known in:

- GEMM implementations
- Reduction kernels

However, applying them to **online softmax attention** requires:

- Careful rescaling logic
- Managing multiple normalization states
- Awareness of register lifetimes

This makes FlashAttention a **non-trivial application** of Split-K.

---

### 7.10 Summary

Partial accumulator writeback (Split-K) directly targets long-lived output accumulators by periodically externalizing partial results.

This approach:
- Reduces persistent VGPR usage
- Preserves numerical correctness
- Trades memory bandwidth for occupancy
- Aligns well with wavefront-64 execution

Split-K is a strong candidate for AMD-optimized FlashAttention variants where register pressure is the primary bottleneck.

---

## 8. Two-Phase Softmax: Breaking the Softmax Dependency Chain

### 8.1 Motivation

In FlashAttention-style kernels, the `online softmax formulation` creates a strict dependency chain:

- The running maximum (`m_i`)
- The normalization term (`l_i`)
- The output accumulator (`acc`)

These quantities must remain live across **all KV blocks** to guarantee numerical correctness.

This dependency is the **root cause** of long-lived registers:

- `m_i`, `l_i` cannot be finalized early
- `acc` must be rescaled whenever a new maximum appears
- All three are updated together in each iteration

The Two-Phase Softmax strategy breaks this dependency chain by **decoupling softmax statistics from value accumulation.**

---

### 8.2 Key Insight

Softmax normalization can be expressed in two logically independent steps:

#### 1. Score statistics
Compute per-query:

- Global maximum over all keys
- Global sum of exponentials

#### 2. Value accumulation
Compute the weighted sum of values using the final normalization

By separating these phases, we eliminate the need for:

- Online rescaling
- Long-lived softmax state
- Persistent accumulator mutation

---

### 8.3 Baseline vs Two-Phase Comparison

| Aspect                 | Online Softmax (Baseline) | Two-Phase Softmax |
| ---------------------- | ------------------------- | ----------------- |
| Softmax state lifetime | Entire KV loop            | Phase-local       |
| Accumulator lifetime   | Entire KV loop            | Phase-local       |
| Register pressure      | High                      | Lower             |
| Memory traffic         | Low                       | Higher            |
| Kernel complexity      | Moderate                  | Higher            |


### 8.4 Phase 1: Score Reduction Kernel

#### Goal:
Compute softmax statistics **without touching V**.

For each query row:

```ini
m = max_j(QKᵀ)
l = Σ_j exp(QKᵀ - m)
```

Characteristics:

- Only Q and K are accessed
- No output accumulator
- No value tensor involved
- Short-lived registers

This phase can be:

- A separate kernel
- A lightweight reduction pass
- Potentially more occupancy-friendly

### 8.5 Phase 2: Value Accumulation Kernel

#### Goal:
Compute the final attention output using precomputed statistics.

For each query row:

```ini
O = Σ_j exp(QKᵀ - m) / l · V
```

Characteristics:

- Softmax normalization constants (`m`, `l`) are read-only
- No need to track running maxima
- Accumulator does not require rescaling
- Accumulator lifetime is limited to this phase only

---

### 8.6 Why This Reduces Register Pressure

Two-Phase Softmax eliminates:

- Long-lived `m_i` and `l_i`
- Online accumulator rescaling
- Dependency between score and value phases

As a result:

- The accumulator is simpler
- Register live ranges are shorter
- VGPR allocation pressure is reduced

This is especially beneficial on AMD GPUs where:

- Wavefront-64 amplifies register usage
- Occupancy is sensitive to peak VGPR count

---

### 8.7 Tradeoffs and Costs

| Cost                     | Impact                                 |
| ------------------------ | -------------------------------------- |
| Extra memory traffic     | Score statistics must be stored        |
| Additional kernel launch | Phase separation                       |
| Reduced fusion           | Less opportunity for instruction reuse |
| Synchronization          | Phase boundary required                |


Two-Phase Softmax is therefore a **throughput vs occupancy tradeoff.**

---

### 8.8 When Two-Phase Softmax Makes Sense

Good candidates:

- Long sequences
- Large head dimensions
- Occupancy-limited kernels
- AMD GPUs with high VGPR sensitivity

Poor candidates:

- Short sequences
- Memory-bound workloads
- Small head dimensions

---

### 8.9 Relationship to Split-K

Split-K and Two-Phase Softmax are **orthogonal**:

- Split-K reduces accumulator lifetime
- Two-Phase Softmax removes accumulator dependencies

They can be:

- Used independently
- Combined for more aggressive VGPR reduction

---

### 8.10 Summary

Two-Phase Softmax removes the loop-carried dependency inherent in online softmax by separating score normalization from value accumulation.

This design:
- Eliminates long-lived softmax state (`m_i`, `l_i`)
- Shortens accumulator live ranges
- Significantly reduces VGPR pressure
- Improves achievable occupancy on wavefront-64 architectures

The tradeoff is increased memory traffic and additional kernel launches, which must be amortized by improved occupancy and latency hiding.

---

## 9. Quantitative Comparison: VGPR vs Occupancy vs Bandwidth

This section compares accumulator lifetime reduction strategies using three primary metrics:

- **VGPR usage** — determines wavefront residency
- **Occupancy** — ability to hide latency
- **Memory bandwidth** — cost of reducing register pressure

All numbers below are **directional**, intended to illustrate scaling behavior rather than exact counts.

---

### 9.1 Baseline: Online Softmax (Reference)

| Metric          | Characteristics                            |
| --------------- | ------------------------------------------ |
| VGPR usage      | Very high (persistent `acc`, `m_i`, `l_i`) |
| Occupancy       | Low (often 2–4 wavefronts / CU)            |
| Bandwidth       | Minimal                                    |
| Latency hiding  | Poor on AMD                                |
| Best suited for | NVIDIA warp-32 GPUs                        |


### 9.2 Smaller Tiles (e.g., BLOCK_M = 64)

| Metric               | Effect                      |
| -------------------- | --------------------------- |
| VGPR usage           | ↓ proportional to tile size |
| Occupancy            | ↑ modest                    |
| Bandwidth            | ↑ due to more tiles         |
| Arithmetic intensity | ↓                           |
| Risk                 | Underutilizing compute      |


This approach is simple but sacrifices compute efficiency.

---

### 9.3 Re-materializing Query Fragments

| Metric     | Effect                   |
| ---------- | ------------------------ |
| VGPR usage | ↓ (shorter Q live range) |
| Occupancy  | ↑                        |
| Bandwidth  | ↑ (extra Q loads)        |
| Complexity | Low                      |
| Risk       | Cache pressure           |

Effective when Q reuse is not dominant.

---

### 9.4 Split-K (Partial Accumulator Writeback)

| Metric          | Effect                |
| --------------- | --------------------- |
| VGPR usage      | ↓↓                    |
| Occupancy       | ↑↑                    |
| Bandwidth       | ↑↑                    |
| Synchronization | Required              |
| Risk            | Memory-bound behavior |

Split-K directly attacks the dominant long-lived accumulator.

### 9.5 Two-Phase Softmax

| Metric          | Effect                    |
| --------------- | ------------------------- |
| VGPR usage      | ↓↓↓                       |
| Occupancy       | ↑↑↑                       |
| Bandwidth       | ↑↑↑                       |
| Kernel launches | 2                         |
| Risk            | Reduced fusion efficiency |

This is the most aggressive register-pressure mitigation.

---

### 9.6 Combined Strategies

| Combination                  | Expected Outcome                        |
| ---------------------------- | --------------------------------------- |
| Smaller tiles + Split-K      | Balanced improvement                    |
| Split-K + Two-Phase          | Maximum occupancy                       |
| Re-materialization + Split-K | Moderate bandwidth, good VGPR reduction |

Combined approaches should be evaluated carefully due to compounded bandwidth cost.

---

## 10. Decision Matrix: Choosing the Right Strategy

| Scenario                      | Recommended Strategy    |
| ----------------------------- | ----------------------- |
| Long sequence, large head dim | Two-Phase Softmax       |
| VGPR-limited, bandwidth-rich  | Split-K                 |
| Small sequences               | Baseline                |
| Moderate sequence, AMD GPU    | Smaller tiles + Split-K |
| Triton prototype              | Re-materialization      |
| Production ROCm kernel        | Split-K or Two-Phase    |

## Conclusion

FlashAttention’s online softmax design fundamentally favors warp-32 execution with abundant register reuse. On wavefront-64 architectures, the same design induces long-lived VGPR pressure that sharply limits occupancy and latency hiding.

This work shows that:
- The performance gap is structural, not incidental
- VGPR pressure can be quantified and traced to specific design choices
- Multiple viable redesigns exist that trade bandwidth for occupancy

No single strategy is universally optimal. Instead, AMD-friendly FlashAttention implementations should adapt kernel structure based on sequence length, head dimension, and hardware constraints.

Future work should focus on hybrid designs and architecture-aware autotuning.
