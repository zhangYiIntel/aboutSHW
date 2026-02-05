# Performance Optimization Summary (Recurrent GDN Kernel)

## Key Optimization Points

- **Sub-group block loads**
  - Use `intel_sub_group_block_read` for `q`, `k`, `v`, and `initial_state` to reduce load instructions and improve coalescing.

- **Per-lane work partitioning**
  - Iterate `j = id_sg_local; j += 16` so each lane processes its own element, avoiding redundant work and enabling efficient SIMD16 execution.

- **Reduce register footprint**
  - Store only per-lane data using arrays of size `K_HEAD_DIMS / 16`, improving occupancy by reducing register pressure.

- **V-blocking**
  - Process multiple V items per workgroup to amortize overhead and better utilize the subgroup.

- **Loop ordering / reuse**
  - Move the V loop inside the SEQ loop to reuse `b_k` and `b_q` loads across multiple V items.

- **Hoist invariants**
  - Precompute base pointers and offsets (`g_ptr`, `beta_ptr`, `out_base`, `kv_base`) to reduce address arithmetic inside hot loops.

- **FMA usage**
  - Use `fma()` for accumulation to improve throughput and numerical stability.

- **Minimize barriers**
  - Avoid unnecessary barriers by relying on subgroup operations and per-lane ownership of data.

- **Avoid bounds checks**
  - Remove per-lane bounds checks when dimensions are guaranteed multiples of 16 to eliminate overhead.

## Outcome

- Bandwidth improved from **~0.8 GB/s** to **~7 GB/s** after applying the optimizations above.
