# recurrent_gated_delta_rule.asm ↔ cm_recurrent_linear.hpp annotation

This note maps key regions in the OpenCL ISA dump to the CM source. The ISA dump is for the OpenCL kernel and does not include CM `LOC` tags, so this is a semantic mapping by operation patterns and message types.

**Sources**
- OpenCL ISA: [opencl/clops/recurrent_gated_delta_rule.asm](opencl/clops/recurrent_gated_delta_rule.asm)
- CM source: [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp)

## 1) Thread and group IDs / stride math
- **OpenCL ISA:** prolog address math (early `mov`, `and`, `add`, `mad`, `shl`) after the declarations.
- **CM source:** `b_idx`, `head_idx`, `head_dim_t_idx` and stride computation in [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L231-L239) and inside `recurrent_linear_attn` (lines ~90–120).

## 2) Initial `h0` loads from `initial_state`
- **OpenCL ISA:** repeated global memory loads (UGM) that fetch 128B chunks; in the OpenCL CSV these show as `send.ugm` with resp-length 4, in ISA they appear as UGM load messages.
- **CM source:** `cm_load_by_row(... initial_state ...)` inside the unrolled loop [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L90-L99).

## 3) `g`/`beta` scalar loads
- **OpenCL ISA:** scalar load messages (UGM) followed by scalar ALU ops.
- **CM source:** `cm_load_by_row<float, IN_OUT_DTYPE, 1>` at [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L104-L108).

## 4) `exp(g)`
- **OpenCL ISA:** math `exp` sequence (approximation + clamps).
- **CM source:** `float g_cur = cm_exp(b_g[0] * log2e);` in [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L164-L165).

## 5) `b_q` / `b_k` vector loads
- **OpenCL ISA:** vector UGM loads feeding the compute section.
- **CM source:**
  - `b_q` load at [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L133-L134)
  - `b_k` load at [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L135-L136)

## 6) Scale `h0 *= g_cur` and `h_k = dot(h0, b_k)`
- **OpenCL ISA:** vector `mul` followed by reduction chains (`add` tree).
- **CM source:** loop with `h0 *= g_cur` and `h_k[i] = cm_sum(...)` at [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L169-L176).

## 7) `delta_full = (b_v - h_k) * b_beta`
- **OpenCL ISA:** scalar `sub`/`mul` using `b_v` and reduction result.
- **CM source:** `delta_full = (b_v - h_k) * b_beta[0];` at [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L183-L183).

## 8) Update `h0 += b_k * delta`
- **OpenCL ISA:** FMA (`mad`) chains across vector lanes.
- **CM source:** `h0 += b_k * detla;` in the second unrolled loop at [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L190-L196).

## 9) Output dot + store
- **OpenCL ISA:** vector dot reduction followed by a store message to output.
- **CM source:** `cur_output[i] = cm_sum(... * b_q);` and `cm_store_by_row(... output ...)` at [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L193-L204).

## 10) Final write-back to `initial_state`
- **OpenCL ISA:** repeated UGM stores at the end of the kernel.
- **CM source:** `cm_store_by_row(... initial_state ...)` at [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp#L206-L214).

---

If you want, I can expand this into a per-address annotation table for the ISA dump (similar to the VISA mapping), but I’ll need a few specific address ranges from the sections you care about (e.g., the first output store or the first `exp` block).