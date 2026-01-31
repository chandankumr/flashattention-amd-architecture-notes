**compare CUDA vs Triton vs ROCm.**

Here is the direct mapping of the **long-lived accumulator pattern** (from FlashAttention's CUDA/CUTE design) onto the provided **Triton implementation** in `flash_attn_triton_amd/fwd_prefill.py`. The core structural similarity is preserved.

### Key locations where the same long-lived accumulator pattern appears

- **Persistent output accumulator (`acc`)** — equivalent to `acc_o` in the CUTE kernel
  - Declared **once**, outside the main KV loop:
    ```python
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=ACCUMULATOR_TYPE)
    ```
  - Lives for the **entire kernel lifetime** of the workgroup (program_id block)
  - Initialized to zero **only once** at the beginning
  - Updated **in-place** inside `_attn_fwd_inner` via repeated `acc += tl.dot(...)` calls
  - Never re-allocated or cleared inside the loop → **very long register/SGPR lifetime**

- **Online softmax state (running max and denominator)** — equivalent to `m` and `l` / `m_i` and `l_i` in original FlashAttention
  - Declared **once**, before entering any KV loop:
    ```python
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=ACCUMULATOR_TYPE)
    l_i = tl.full([BLOCK_M], 1.0, dtype=ACCUMULATOR_TYPE)
    ```
  - These are **per-query-row** persistent statistics
  - Updated **in every iteration** of the KV loop inside `_attn_fwd_inner`:
    - `m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))`
    - `m_i = m_ij`
    - `l_i = l_i * alpha + l_ij`
  - Must remain live across **all** KV blocks to enable correct online rescaling → long lifetime

- **Main loop over KV blocks** — where the accumulator lifetime becomes problematic
  - The function `_attn_fwd_inner` contains the **loop over start_n**:
    ```python
    for start_n in range(block_min, block_max, BLOCK_N):
        ...
        acc = acc * alpha[:, None]               # rescale old accumulator
        acc += tl.dot(p.to(v.type.element_ty), v)   # accumulate new contribution
        ...
    ```
  - This loop is called **twice** in the outer kernel (once for full blocks, once for masked blocks), but `acc`, `m_i`, `l_i` are **passed in and out** and keep their values:
    ```python
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, ...)
    ```
  - All three tensors (`acc`, `m_i`, `l_i`) remain **live and mutated across the full sequence of KV tiles**

- **Query fragment (`q`)** — partial analog to Q-in-registers mode
  - Loaded **once** before the KV loops:
    ```python
    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)
    ```
  - Re-used in **every iteration** of the inner loop (`tl.dot(q, k)`)
  - Lives for the entire duration → another long-lived vector per thread

- **Why this is particularly painful on AMD CDNA / wavefront-64**
  - `acc` is `[BLOCK_M, BLOCK_DMODEL]` — typically 128×128 or 128×64 in the autotune configs → very large when distributed across 64-wide wavefronts
  - Each lane holds ~ (BLOCK_DMODEL / wavefront_size) × BLOCK_M elements of `acc` (frequently 16–32 fp32 values per lane)
  - `m_i` and `l_i` add ~2×BLOCK_M scalars per lane
  - Total → **high pressure on VGPRs** (and sometimes SGPRs for broadcasting)
  - Wavefront-64 means **twice as many lanes** per scheduling unit compared to warp-32 → register file consumption scales poorly, often dropping occupancy more aggressively than on NVIDIA
  - Triton compiler currently has **less aggressive register packing / spilling heuristics** than nvcc+PTX → more likely to spill these long-lived vectors to LDS or global memory

### Summary — exact analogy table

| Concept (CUDA/CUTE FlashAttention) | Equivalent in Triton fwd_prefill.py              | Lifetime scope                  | Why long-lived?                          |
|-------------------------------------|--------------------------------------------------|----------------------------------|-------------------------------------------|
| `acc_o` (output accumulator)       | `acc`                                            | Entire kernel (WG)              | In-place += across all KV blocks         |
| `m`, `l` (online softmax state)    | `m_i`, `l_i`                                     | Entire kernel (WG)              | Must track global max & sum-of-exp       |
| Q in registers                     | `q` (loaded once)                                | Entire kernel                   | Reused for every K block                 |
| Loop over KV blocks                | `for start_n in range(block_min, block_max, BLOCK_N)` inside `_attn_fwd_inner` | —                               | Accumulators mutated but never killed    |

In short: the **online-softmax + incremental-accumulation** design pattern is **fully preserved** in this Triton port — including the exact same long register lifetimes for `acc`, `m_i`, and `l_i`. This is why register pressure concerns carry over almost unchanged from the CUDA version to AMD GPUs when using this Triton kernel.