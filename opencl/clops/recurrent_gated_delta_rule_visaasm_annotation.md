# recurrent_gated_delta_rule.visaasm ↔ cm_recurrent_linear.hpp annotation

This note maps key VISA asm regions to the CM source in [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp). The VISA has `LOC` tags that correspond to source line numbers in the CM file, and helper functions (`cm_load_by_row`, `cm_store_by_row`) are inlined, so many `lsc_load.ugm`/`lsc_store.ugm` lines point at helper lines rather than call sites.

## Entry + kernel dispatch
- **VISA:** Function prolog and call to template implementation in [opencl/clops/recurrent_gated_delta_rule.visaasm](opencl/clops/recurrent_gated_delta_rule.visaasm#L760-L790).
- **CM:** `cm_group_id`/`cm_local_id` and template call in [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L231-L257).

## Initial `h0` (initial_state) loads
- **VISA:** `lsc_load.ugm ... d32x64t` blocks starting at LOC 27/28 in [opencl/clops/recurrent_gated_delta_rule.visaasm](opencl/clops/recurrent_gated_delta_rule.visaasm#L770-L910).
- **CM helper:** `cm_load_by_row` (same-type) uses `cm_load<uint, 64>` when `unified_N == 128` in [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L22-L34).
- **CM call site:** `cm_load_by_row(... initial_state ...)` inside the unrolled `h0` load loop in [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L90-L99).

## q/k/v loads inside seq loop
- **VISA:** More `lsc_load.ugm ... d32x64t` sequences with LOC 27/28 (helper) plus address arithmetic around LOC 94–98 for stride setup in [opencl/clops/recurrent_gated_delta_rule.visaasm](opencl/clops/recurrent_gated_delta_rule.visaasm#L770-L930).
- **CM call sites:**
  - `b_q` load: [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L133-L134)
  - `b_k` load: [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L135-L136)
  - `b_v` load: [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L146-L148)

## exp(g) and `h0` scaling
- **VISA:** math + reduction sequences around LOC 164/168/190 in [opencl/clops/recurrent_gated_delta_rule.visaasm](opencl/clops/recurrent_gated_delta_rule.visaasm#L1560-L1660).
- **CM:** `g_cur = cm_exp(...)` and the unrolled `h0 *= g_cur` loop in [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L164-L176).

## `h_k` dot and `delta_full`
- **VISA:** repeated `mad`/`mul`/`add` chains (vector dot + reduction) in [opencl/clops/recurrent_gated_delta_rule.visaasm](opencl/clops/recurrent_gated_delta_rule.visaasm#L1560-L1680).
- **CM:** `h_k[i] = cm_sum(h0 * b_k)` and `delta_full = (b_v - h_k) * b_beta` in [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L171-L186).

## Output dot + store
- **VISA:** `lsc_store.ugm` to `bti(0x7)` for output (vector length 16) at LOC 44 in [opencl/clops/recurrent_gated_delta_rule.visaasm](opencl/clops/recurrent_gated_delta_rule.visaasm#L1636-L1645).
- **CM helper:** `cm_store_by_row` (same-type) in [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L36-L45).
- **CM call site:** output store in [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L200-L204).

## Final write-back of `initial_state`
- **VISA:** batch of `lsc_store.ugm ... d32x64t` at LOC 41/42 in [opencl/clops/recurrent_gated_delta_rule.visaasm](opencl/clops/recurrent_gated_delta_rule.visaasm#L1646-L1706).
- **CM helper:** `cm_store_by_row` same-type in [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L36-L45).
- **CM call site:** final `cm_store_by_row(... initial_state ...)` in [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L206-L214).

## Notes
- LOC numbers in VISA map to CM source line numbers. The load/store helpers are inlined, so many `lsc_load.ugm`/`lsc_store.ugm` reference helper lines instead of the call sites.
- The unrolled `v_head_dim_per_t` loop expands to many repeated blocks of `mad`/`mul`/`add` that correspond to dot-products and reductions in the CM code.
