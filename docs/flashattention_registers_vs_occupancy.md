# Register Pressure vs Occupancy: FlashAttention on AMD GPUs

This note summarizes the dominant sources of long-lived register pressure in FlashAttention forward kernels and explains why these effects translate more directly into occupancy loss on AMD GPUs than on NVIDIA architectures.

The focus is on *structural causes* rather than implementation bugs.

---

## Long-Lived Register Pressure Sources in FlashAttention

### Output Accumulator (`acc_o`)

The dominant source of register pressure is the FP32 output accumulator, partitioned as a `(kBlockM × kHeadDim)` fragment across threads. This accumulator is initialized once and incrementally updated in every key–value block via `gemm_rs`, accumulating the weighted values from `V`.

Because FlashAttention uses an **online softmax formulation**, the output accumulator must remain live for the entire KV traversal. This prevents early write-back or register reuse and results in a large number of registers being held per thread for the full kernel lifetime.

---

### Online Softmax State (Running Max and Normalization)

The softmax implementation maintains per-row running maxima and normalization terms that are updated in each call to `softmax_rescale_o`. These values are required to correctly rescale both intermediate scores and the output accumulator whenever a new maximum score is encountered.

As a result, this state must persist across all iterations of the KV loop, contributing additional long-lived registers whose lifetimes match that of the output accumulator.

---

### Query Fragments Held in Registers

In configurations where `Is_Q_in_regs` is enabled, tiled query fragments are retained in registers across the entire KV loop to avoid repeated shared-memory loads. While this improves arithmetic efficiency, it further extends register lifetimes and compounds overall register pressure.

---

### Intermediate Score Accumulator (`acc_s`)

Although the intermediate score accumulator is cleared and reused on each iteration, its large temporary footprint during the GEMM, masking, and softmax phases increases *peak* register demand. This peak overlaps with the simultaneously live output accumulator and softmax state, further stressing the register file.

---

## Architectural Amplification on AMD GPUs

### Wavefront Size Effects

AMD GPUs execute 64-thread wavefronts, compared to 32-thread warps on NVIDIA. For the same per-thread register usage, this doubles register file consumption per scheduling unit. As a result, kernels with large, long-lived accumulators experience sharper occupancy cliffs once VGPR thresholds are crossed.

---

### Scheduling Granularity and Latency Hiding

With fewer independently schedulable wavefronts per compute unit, AMD GPUs have less opportunity to hide memory and pipeline latency when register pressure already limits residency. This amplifies the performance sensitivity of accumulator-heavy kernels.

---

### Compiler Allocation Tradeoffs

Differences in register allocation heuristics, spilling behavior, and instruction scheduling between ROCm and CUDA further influence effective register usage in deeply templated, accumulator-heavy kernels such as FlashAttention. In practice, this often makes the inherent register demands of the online softmax design more performance-critical on AMD architectures.

---

## Takeaway

FlashAttention’s performance model deliberately trades global memory bandwidth for persistent on-chip state. On AMD GPUs, wavefront-64 execution and occupancy constraints make this tradeoff more visible. Understanding the register lifetime structure is a prerequisite for designing AMD-optimized variants rather than attempting micro-level tuning.
