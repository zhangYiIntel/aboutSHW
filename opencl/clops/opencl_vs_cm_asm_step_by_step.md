# OpenCL asm vs CM VISA: step-by-step comparison

This table aligns major steps in the algorithm and shows where they appear in the OpenCL asm dump and the CM VISA. It is a semantic mapping (algorithm step → asm region) rather than a strict one-to-one instruction match.

**Sources**
- OpenCL asm: [opencl/clops/good_opencl.csv](opencl/clops/good_opencl.csv)
- CM VISA: [opencl/clops/recurrent_gated_delta_rule.visaasm](opencl/clops/recurrent_gated_delta_rule.visaasm)
- CM source: [opencl/clops/cm_recurrent_linear.hpp](opencl/clops/cm_recurrent_linear.hpp)
- OpenCL source: [opencl/clops/recurrent_linear.py](opencl/clops/recurrent_linear.py)

## Step-by-step table

| Step | Algorithm stage | OpenCL asm evidence | CM VISA evidence | Notes |
|---|---|---|---|---|
| 1 | Prolog: IDs/strides | Address/stride math around early block (0x40–0x2b0) | Function prolog + address math at start of BB_1 (LOC 94–98) | Both compute `b`, `h`, `v_block`, pointer bases; CM uses LOC tags. |
| 2 | Initial state load (`h0`) | Multiple `send.ugm` loads with resp-length 4 (0x3e0–0x450) | Repeated `lsc_load.ugm ... d32x64t` at LOC 27/28 in [opencl/clops/recurrent_gated_delta_rule.visaasm](opencl/clops/recurrent_gated_delta_rule.visaasm#L770-L910) | OpenCL uses UGM message; CM uses LSC path. Both load 128B chunks. |
| 3 | `g`/`beta` load | `send.ugm` scalar loads near 0x530–0x5b8 | `lsc_load.ugm` scalar loads near LOC 106–108 (inline cm_load_by_row for N=1) | Both load scalar `g`, `beta`. |
| 4 | `exp(g)` | `math.exp` sequence 0x5f0–0x690 | `cm_exp` expansion around LOC 164–168 | Both use exp approximation; scheduling differs. |
| 5 | Load `b_k`/`b_q` | Block reads via `send.ugm` (0x3e0–0x450, 0x530–0x5b8) | `lsc_load.ugm ... d32x64t` at LOC 27/28 (same helper as `h0`) | Same vector load size. |
| 6 | Scale `h0 *= g_cur` | `mul` instructions using exp result (0x6a8–0x6f8) | `mul` chains near LOC 168/190 | Both multiply vector by scalar. |
| 7 | `h_k = dot(h0, b_k)` | Reduction tree 0x6c8–0x768 | `mad/mul` + reduction chain around LOC 190 | Both implement vector dot + reduction. |
| 8 | `b_v = (v - h_k) * beta` | Scalar load + subtract + mul around 0x7c8–0x7f8 | Scalar ops tied to `b_v` in LOC 191 block | Same scalar sequence. |
| 9 | Update `h0 += b_k * b_v` | `mad` sequences 0x7d8–0x820 and later | `mad` chains around LOC 190/191 | Both FMA update. |
|10 | Output dot `out_acc = dot(h0, b_q)` | Reduction tree 0x828–0x8d8 | Similar `mad`/reduce chain LOC 190/191 | Same compute but register allocation differs. |
|11 | Output store | `send.ugm` store to output (0x920–0x938) | `lsc_store.ugm ... d32x16t` at LOC 44 in [opencl/clops/recurrent_gated_delta_rule.visaasm](opencl/clops/recurrent_gated_delta_rule.visaasm#L1636-L1645) | CM output store uses LSC; OpenCL uses UGM. |
|12 | Final state writeback | `send.ugm` stores near end | `lsc_store.ugm ... d32x64t` at LOC 41/42 in [opencl/clops/recurrent_gated_delta_rule.visaasm](opencl/clops/recurrent_gated_delta_rule.visaasm#L1646-L1706) | Both store 128B chunks. |

## Which is better?

It depends on the target and driver:
- **CM VISA (LSC path)** can be better on newer Intel GPUs where LSC is optimized and the compiler can fuse cm_load_by_row / cm_store_by_row patterns effectively.
- **OpenCL UGM path** can be competitive or better on some driver versions, especially if the OpenCL compiler schedules subgroup ops more aggressively for your exact kernel shape.

The best choice is empirical: measure BW and latency on your specific GPU + driver. If your current results show CM faster or more stable, keep CM; if OpenCL outperforms with your subgroup block reads, prefer the OpenCL path.
