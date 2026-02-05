# cl_kernel_sources = r'''
# float dot_product(float* a, float* b, size_t n) {
#     float result = 0.0f;
#     for (size_t i = 0; i < n; i++) {
#         result += a[i] * b[i];
#     }
#     return result;
# }

# void scale(float *a, float scale, size_t n) {
#     for (size_t i = 0; i < n; i++) {
#         a[i] = a[i] * scale;
#     }
# }

# void add(float *a, float* b, size_t n) {
#     for (size_t i = 0; i < n; i++) {
#         a[i] += b[i];
#     }
# }

# __kernel void recurrent_gated_delta_rule(__global float * q, 
#     __global float * k, 
#     __global float * v, 
#     __global float * g,
#     __global float * beta,
#     __global float * initial_state,
#     __global float * output) {
#     int b = get_global_id(0);
#     int h = get_global_id(1);
#     int i_v = get_group_id(2);
#     int BATCH_STRIDE = BATCH_NUM * K_HEAD_NUMS * SEQ_LEN;
#     int HEAD_STRIDE = SEQ_LEN * K_HEAD_DIMS;
#     __global float* q_ptr = q + b * BATCH_STRIDE + h * HEAD_STRIDE;
#     __global float* k_ptr = k + b * BATCH_STRIDE + h * HEAD_STRIDE;
#     __global float* v_ptr = v + b * BATCH_STRIDE + h * HEAD_STRIDE;
#     float init_state[K_HEAD_DIMS] = {0};
#     float b_k[K_HEAD_DIMS] = {0};
#     float b_q[K_HEAD_DIMS] = {0};
#     for (int j = 0; j < K_HEAD_DIMS; j++) {
#         init_state[j] = initial_state[b * K_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS + j];
#     }
#     for (int i = 0; i < SEQ_LEN; i++) {
#         float b_g = g[b * K_HEAD_NUMS * SEQ_LEN + h * SEQ_LEN + i];
#         float b_beta = beta[b * K_HEAD_NUMS * SEQ_LEN + h * SEQ_LEN + i];
#         b_g = exp(b_g);
#         for (int j = 0; j < K_HEAD_DIMS; j++) {
#             b_k[j] = k_ptr[i * K_HEAD_DIMS  + j];
#             b_q[j] = q_ptr[i * K_HEAD_DIMS  + j];
#         }
#         // h0 * g
#         scale(init_state, b_g, K_HEAD_DIMS);
#         float h_k = dot_product(init_state, b_k, K_HEAD_DIMS);
#         float b_v = v_ptr[i_v + i * K_HEAD_DIMS];
#         b_v -= h_k;
#         // b_v * b_k
#         b_v *= b_beta;
#         scale(b_k, b_v, K_HEAD_DIMS);
#         // h = h0 + update
#         add(init_state, b_k, K_HEAD_DIMS);
#         float b_output  = dot_product(init_state, b_q, K_HEAD_DIMS);
#         output[b * K_HEAD_NUMS * SEQ_LEN * K_HEAD_DIMS + h * SEQ_LEN * K_HEAD_DIMS + i * K_HEAD_DIMS + i_v] = b_output;
#         //printf("b %d h %d i_v %d seq %i output %f init_state %f\n", b, h, i_v, i, b_output, init_state[0]);
#     }

# /*
#     printf("b %d h %d get_sub_group_local_id %d\n", b, h, get_sub_group_local_id());
#     for (size_t i = 0; i < T; i++) {
#         float partial = 0.0f;
        
#         for (int j = lid; j < 16; j += lsize) {
#             half a = as_half(intel_sub_group_block_read_us((__global ushort*)(q_ptr + j + i *16)));
#             partial += a;
#         }
#         float sub_sum = sub_group_reduce_add(partial);

#         if (get_sub_group_local_id() == 0) {
#             printf("batch %d head %d q %zu value %f\n", b, h, i, sub_sum);
#         }
#     }
# */
# }

# '''

cm_kernel_sources = r'''
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
float dot_product(float* a, float* b, size_t n) {
    float result = 0.0f;
    for (size_t i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

void scale(float *a, float scale, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] = a[i] * scale;
    }
}

void add(float *a, float* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] += b[i];
    }
}
#define V_BLOCK_SIZE 4
float sg_read_f(__global const float* p) {
    return as_float(intel_sub_group_block_read((__global const uint*)p));
}
float2 sg_read2_f(__global const float* p) {
    return as_float2(intel_sub_group_block_read2((__global const uint*)p));
}
float8 sg_read8_f(__global const float* p) {
    return as_float8(intel_sub_group_block_read8((__global const uint*)p));
}
half2 sg_read2_h(__global const half* p) {
    return as_half2(intel_sub_group_block_read((__global const uint*)p));
}
half8 sg_read8_h(__global const half* p) {
    return as_half8(intel_sub_group_block_read4((__global const uint*)p));
}
void sg_write2_f(__global float* p, float2 v) {
    intel_sub_group_block_write2((__global uint*)p, as_uint2(v));
}
void sg_write8_f(__global float* p, float8 v) {
    intel_sub_group_block_write8((__global uint*)p, as_uint8(v));
}
void sg_write2_h(__global half* p, half2 v) {
    intel_sub_group_block_write((__global uint*)p, as_uint(v));
}
void sg_write8_h(__global half* p, half8 v) {
    intel_sub_group_block_write4((__global uint*)p, as_uint4(v));
}
float sum2(float2 v) {
    return v.s0 + v.s1;
}
float sum8(float8 v) {
    return v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7;
}
void sg_write_f(__global float* p, float v) {
    intel_sub_group_block_write((__global uint*)p, as_uint(v));
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void recurrent_gated_delta_rule(__global float * q, 
    __global float * k, 
    __global float * v, 
    __global float * g,
    __global float * beta,
    __global float * initial_state,
    __global float * output) {
    int b = get_global_id(0);
    int gid1 = get_global_id(1);
    int id_local = get_local_id(2);
    int BATCH_STRIDE = BATCH_NUM * K_HEAD_NUMS * SEQ_LEN;
    int HEAD_STRIDE = SEQ_LEN * K_HEAD_DIMS;
    int v_blocks = (K_HEAD_DIMS + V_BLOCK_SIZE - 1) / V_BLOCK_SIZE;
    int h = gid1 / v_blocks;
    int v_block_id = gid1 - h * v_blocks;
    int i_v_base = v_block_id * V_BLOCK_SIZE;
    __global const float* q_ptr = q + b * BATCH_STRIDE + h * HEAD_STRIDE;
    __global const float* k_ptr = k + b * BATCH_STRIDE + h * HEAD_STRIDE;
    __global const float* v_ptr = v + b * BATCH_STRIDE + h * HEAD_STRIDE;
    __global const float* g_ptr = g + b * K_HEAD_NUMS * SEQ_LEN + h * SEQ_LEN;
    __global const float* beta_ptr = beta + b * K_HEAD_NUMS * SEQ_LEN + h * SEQ_LEN;
    int out_base = b * K_HEAD_NUMS * SEQ_LEN * K_HEAD_DIMS + h * SEQ_LEN * K_HEAD_DIMS;
#if (K_HEAD_DIMS == 128)
    float8 init_state[V_BLOCK_SIZE];
    float8 b_k;
    float8 b_q;
#elif (K_HEAD_DIMS % 32) == 0
    float2 init_state[V_BLOCK_SIZE][K_HEAD_DIMS / 32];
    float2 b_k[K_HEAD_DIMS / 32];
    float2 b_q[K_HEAD_DIMS / 32];
#else
    float init_state[V_BLOCK_SIZE][K_HEAD_DIMS / 16] = {0};
    float b_k[K_HEAD_DIMS / 16] = {0};
    float b_q[K_HEAD_DIMS / 16] = {0};
#endif
    int id_sg_local = get_sub_group_local_id();
    for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
        int i_v = i_v_base + iv;
        int init_base = b * K_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS;
#if (K_HEAD_DIMS == 128)
    init_state[iv] = sg_read8_f(initial_state + init_base);
#elif (K_HEAD_DIMS % 32) == 0
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
            int idx = j >> 5;
            init_state[iv][idx] = sg_read2_f(initial_state + init_base + (j - id_sg_local));
        }
#else
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += 16) {
            int idx = j >> 4;
            float val = sg_read_f(initial_state + init_base + (j - id_sg_local));
            init_state[iv][idx] = val;
        }
#endif
    }
    int kv_base = 0;
    int out_i_base = out_base;
    for (int i = 0; i < SEQ_LEN; i++, kv_base += K_HEAD_DIMS, out_i_base += K_HEAD_DIMS) {
        float b_g = g_ptr[i];
        float b_beta = beta_ptr[i];
        b_g = exp(b_g);
    #if (K_HEAD_DIMS == 128)
            b_k = sg_read8_f(k_ptr + kv_base);
            b_q = sg_read8_f(q_ptr + kv_base);
    #elif (K_HEAD_DIMS % 32) == 0
        #pragma unroll
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
            int idx = j >> 5;
            b_k[idx] = sg_read2_f(k_ptr + kv_base + (j - id_sg_local));
            b_q[idx] = sg_read2_f(q_ptr + kv_base + (j - id_sg_local));
        }
    #else
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += 16) {
            int idx = j >> 4;
            b_k[idx] = sg_read_f(k_ptr + kv_base + (j - id_sg_local));
            b_q[idx] = sg_read_f(q_ptr + kv_base + (j - id_sg_local));
        }
    #endif
        for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
            int i_v = i_v_base + iv;
#if (K_HEAD_DIMS == 128)
            init_state[iv] *= b_g;
            float hk_acc = sum8(init_state[iv] * b_k);
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base = kv_base + (i_v & ~15);
            int v_lane = i_v & 15;
            float v_val = as_float(intel_sub_group_block_read((__global const uint*)(v_ptr + v_base)));
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            b_v *= b_beta;
            init_state[iv] = fma(b_k, (float8)(b_v), init_state[iv]);

            float out_acc = sum8(init_state[iv] * b_q);
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = out_acc;
            }
#elif (K_HEAD_DIMS % 32) == 0
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                init_state[iv][idx] *= b_g;
            }
            float hk_acc = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                hk_acc += sum2(init_state[iv][idx] * b_k[idx]);
            }
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base = kv_base + (i_v & ~15);
            int v_lane = i_v & 15;
            float v_val = as_float(intel_sub_group_block_read((__global const uint*)(v_ptr + v_base)));
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            b_v *= b_beta;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                init_state[iv][idx] = fma(b_k[idx], (float2)(b_v), init_state[iv][idx]);
            }

            float out_acc = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                out_acc += sum2(init_state[iv][idx] * b_q[idx]);
            }
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = out_acc;
            }
#else
            // h0 * g
            #pragma unroll
            for (size_t n = id_sg_local; n < K_HEAD_DIMS; n+= 16) {
                int idx = n >> 4;
                init_state[iv][idx] *= b_g;
            }
            float hk_acc = 0.0f;
            #pragma unroll
            for (size_t n = id_sg_local; n < K_HEAD_DIMS; n+= 16) {
                int idx = n >> 4;
                hk_acc = fma(init_state[iv][idx], b_k[idx], hk_acc);
            }
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base = kv_base + (i_v & ~15);
            int v_lane = i_v & 15;
            float v_val = as_float(intel_sub_group_block_read((__global const uint*)(v_ptr + v_base)));
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            // b_v * b_k
            b_v *= b_beta;
            // h0 = h0 + b_k * b_v;
            #pragma unroll
            for (size_t n = id_sg_local; n < K_HEAD_DIMS; n+= 16) {
                int idx = n >> 4;
                init_state[iv][idx] = fma(b_k[idx], b_v, init_state[iv][idx]);
            };
            // float b_output  = dot_product(init_state, b_q, K_HEAD_DIMS);
            float out_acc = 0.0f;
            #pragma unroll
            for (size_t n = id_sg_local; n < K_HEAD_DIMS; n+= 16) {
                int idx = n >> 4;
                out_acc = fma(init_state[iv][idx], b_q[idx], out_acc);
            }
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = out_acc;
                // printf("b %d h %d i_v %d seq %i output %f init_state %f\n", b, h, i_v, i, out_acc, init_state[0]);    
            }
#endif
        }
    }
    for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
        int i_v = i_v_base + iv;
        int init_base = b * K_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS;
#if (K_HEAD_DIMS == 128)
    sg_write8_f(initial_state + init_base, init_state[iv]);
#elif (K_HEAD_DIMS % 32) == 0
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
        int idx = j >> 5;
        sg_write2_f(initial_state + init_base + (j - id_sg_local), init_state[iv][idx]);
    }
#else
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += 16) {
            int idx = j >> 4;
            sg_write_f(initial_state + init_base + (j - id_sg_local), init_state[iv][idx]);
        }
#endif
    }

/*
    printf("b %d h %d get_sub_group_local_id %d\n", b, h, get_sub_group_local_id());
    for (size_t i = 0; i < T; i++) {
        float partial = 0.0f;
        
        for (int j = lid; j < 16; j += lsize) {
            half a = as_half(intel_sub_group_block_read_us((__global ushort*)(q_ptr + j + i *16)));
            partial += a;
        }
        float sub_sum = sub_group_reduce_add(partial);

        if (get_sub_group_local_id() == 0) {
            printf("batch %d head %d q %zu value %f\n", b, h, i, sub_sum);
        }
    }
*/
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void recurrent_gated_delta_rule_f16(__global half * q,
    __global half * k,
    __global half * v,
    __global half * g,
    __global half * beta,
    __global half * initial_state,
    __global half * output) {
    int b = get_global_id(0);
    int gid1 = get_global_id(1);
    int id_local = get_local_id(2);
    int BATCH_STRIDE = BATCH_NUM * K_HEAD_NUMS * SEQ_LEN;
    int HEAD_STRIDE = SEQ_LEN * K_HEAD_DIMS;
    int v_blocks = (K_HEAD_DIMS + V_BLOCK_SIZE - 1) / V_BLOCK_SIZE;
    int h = gid1 / v_blocks;
    int v_block_id = gid1 - h * v_blocks;
    int i_v_base = v_block_id * V_BLOCK_SIZE;
    __global const half* q_ptr = q + b * BATCH_STRIDE + h * HEAD_STRIDE;
    __global const half* k_ptr = k + b * BATCH_STRIDE + h * HEAD_STRIDE;
    __global const half* v_ptr = v + b * BATCH_STRIDE + h * HEAD_STRIDE;
    __global const half* g_ptr = g + b * K_HEAD_NUMS * SEQ_LEN + h * SEQ_LEN;
    __global const half* beta_ptr = beta + b * K_HEAD_NUMS * SEQ_LEN + h * SEQ_LEN;
    int out_base = b * K_HEAD_NUMS * SEQ_LEN * K_HEAD_DIMS + h * SEQ_LEN * K_HEAD_DIMS;
 #if (K_HEAD_DIMS == 128)
     float8 init_state[V_BLOCK_SIZE];
     float8 b_k;
     float8 b_q;
 #elif (K_HEAD_DIMS % 32) == 0
     float2 init_state[V_BLOCK_SIZE][K_HEAD_DIMS / 32];
     float2 b_k[K_HEAD_DIMS / 32];
     float2 b_q[K_HEAD_DIMS / 32];
 #else
     float init_state[V_BLOCK_SIZE][(K_HEAD_DIMS + 15) / 16] = {0};
     float b_k[(K_HEAD_DIMS + 15) / 16] = {0};
     float b_q[(K_HEAD_DIMS + 15) / 16] = {0};
 #endif
    int id_sg_local = get_sub_group_local_id();

    for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
        int i_v = i_v_base + iv;
        int init_base = b * K_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS;
 #if (K_HEAD_DIMS == 128)
        half8 h8 = sg_read8_h(initial_state + init_base);
        init_state[iv] = convert_float8(h8);
 #elif (K_HEAD_DIMS % 32) == 0
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
            int idx = j >> 5;
            half2 h2 = sg_read2_h(initial_state + init_base + (j - id_sg_local));
            init_state[iv][idx] = convert_float2(h2);
        }
 #else
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += 16) {
            int idx = j >> 4;
            init_state[iv][idx] = convert_float(initial_state[init_base + j]);
        }
 #endif
    }

    int kv_base = 0;
    int out_i_base = out_base;
    for (int i = 0; i < SEQ_LEN; i++, kv_base += K_HEAD_DIMS, out_i_base += K_HEAD_DIMS) {
         float b_g = exp(convert_float(g_ptr[i]));
         float b_beta = convert_float(beta_ptr[i]);

     #if (K_HEAD_DIMS == 128)
         b_k = convert_float8(sg_read8_h(k_ptr + kv_base));
         b_q = convert_float8(sg_read8_h(q_ptr + kv_base));
     #elif (K_HEAD_DIMS % 32) == 0
         #pragma unroll
         for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
             int idx = j >> 5;
             b_k[idx] = convert_float2(sg_read2_h(k_ptr + kv_base + (j - id_sg_local)));
             b_q[idx] = convert_float2(sg_read2_h(q_ptr + kv_base + (j - id_sg_local)));
         }
     #else
         for (int j = id_sg_local; j < K_HEAD_DIMS; j += 16) {
             int idx = j >> 4;
             b_k[idx] = convert_float(k_ptr[kv_base + j]);
             b_q[idx] = convert_float(q_ptr[kv_base + j]);
         }
     #endif

        for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
            int i_v = i_v_base + iv;
 #if (K_HEAD_DIMS == 128)
            init_state[iv] *= b_g;
            float hk_acc = sum8(init_state[iv] * b_k);
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base = kv_base + (i_v & ~15);
            int v_lane = i_v & 15;
            half v_val_h = as_half(intel_sub_group_block_read_us((__global const ushort*)(v_ptr + v_base)));
            float v_val = convert_float(v_val_h);
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            b_v *= b_beta;
            init_state[iv] = fma(b_k, (float8)(b_v), init_state[iv]);

            float out_acc = sum8(init_state[iv] * b_q);
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = convert_half_rte(out_acc);
            }
 #elif (K_HEAD_DIMS % 32) == 0
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                init_state[iv][idx] *= b_g;
            }
            float hk_acc = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                hk_acc += sum2(init_state[iv][idx] * b_k[idx]);
            }
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base = kv_base + (i_v & ~15);
            int v_lane = i_v & 15;
            half v_val_h = as_half(intel_sub_group_block_read_us((__global const ushort*)(v_ptr + v_base)));
            float v_val = convert_float(v_val_h);
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            b_v *= b_beta;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                init_state[iv][idx] = fma(b_k[idx], (float2)(b_v), init_state[iv][idx]);
            }

            float out_acc = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                out_acc += sum2(init_state[iv][idx] * b_q[idx]);
            }
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = convert_half_rte(out_acc);
            }
 #else
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 16) {
                int idx = j >> 4;
                init_state[iv][idx] *= b_g;
            }
            float hk_acc = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 16) {
                int idx = j >> 4;
                hk_acc = fma(init_state[iv][idx], b_k[idx], hk_acc);
            }
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            float b_v = convert_float(v_ptr[kv_base + i_v]);
            b_v -= hk_acc;
            b_v *= b_beta;

            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 16) {
                int idx = j >> 4;
                init_state[iv][idx] = fma(b_k[idx], b_v, init_state[iv][idx]);
            }

            float out_acc = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 16) {
                int idx = j >> 4;
                out_acc = fma(init_state[iv][idx], b_q[idx], out_acc);
            }
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = convert_half_rte(out_acc);
            }
 #endif
        }
    }

    for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
        int i_v = i_v_base + iv;
        int init_base = b * K_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS;
 #if (K_HEAD_DIMS == 128)
        half8 h8 = convert_half8_rte(init_state[iv]);
        sg_write8_h(initial_state + init_base, h8);
 #elif (K_HEAD_DIMS % 32) == 0
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
            int idx = j >> 5;
            half2 h2 = convert_half2_rte(init_state[iv][idx]);
            sg_write2_h(initial_state + init_base + (j - id_sg_local), h2);
        }
 #else
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += 16) {
            int idx = j >> 4;
            initial_state[init_base + j] = convert_half_rte(init_state[iv][idx]);
        }
 #endif
    }
}
'''

from . import cl
import numpy as np
from .utils import *
import torch.nn.functional as F

def get_abs_err(x, y):
    return (x.detach()-y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach()-y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    error_rate = get_err_ratio(ref, tri)
    print(abs_atol, err_atol)
    if abs_atol <= err_atol:
        return "Good Result"
    else:
        print(f"error_rate {error_rate} ratio {ratio}")
        assert error_rate < ratio, msg

def recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    # q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float16), [q, k, v, beta, g])
    B, H, T, K, V = *k.shape, v.shape[-1]
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    beta = beta.to(torch.float32)
    g = g.to(torch.float32)
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state.transpose(-2, -1).contiguous()
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        # print("g[:, :, i].exp() ", g[:, :, i].exp())
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        h = h.clone()
        b_beta = beta[:, :, i]
        # print("beta ", beta)
        # print((h.clone() * b_k[..., None]).sum(-2))
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        # print("bv ", b_v)
        b_v = b_v * b_beta[..., None]
        # print("bk*bv ", b_k.unsqueeze(-1) * b_v.unsqueeze(-2))
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)
    if not output_final_state:
        h = None
    else:
        h = h.transpose(-2, -1).contiguous()
    # o = o.transpose(1, 2).contiguous()
    return o, h

class RecurrentGDN:
    def __init__(self, B, H, T, K, V):
        self.B = B
        self.H = H
        self.T = T
        self.K = K
        self.V = V
        self.kernels = kernel_cache(cm_kernel_sources, f"-DBATCH_NUM={self.B} -DK_HEAD_NUMS={self.H} -DSEQ_LEN={self.T} -DK_HEAD_DIMS={self.K}", "./dump/")
    
    def __call__(self, q, k, v, g, beta, hidden_states, o):
        v_block_size = 4
        v_blocks = (self.V + v_block_size - 1) // v_block_size
        self.kernels.enqueue("recurrent_gated_delta_rule",
                            [self.B, self.H * v_blocks, 16],
                            [1, 1, 16],
                            q,
                            k,
                            v,
                            g,
                            beta,
                            hidden_states,
                            o
                            )

class RecurrentGDNHalf:
    def __init__(self, B, H, T, K, V):
        self.B = B
        self.H = H
        self.T = T
        self.K = K
        self.V = V
        self.kernels = kernel_cache(cm_kernel_sources, f"-DBATCH_NUM={self.B} -DK_HEAD_NUMS={self.H} -DSEQ_LEN={self.T} -DK_HEAD_DIMS={self.K}", "./dump/")

    def __call__(self, q, k, v, g, beta, hidden_states, o):
        v_block_size = 4
        v_blocks = (self.V + v_block_size - 1) // v_block_size
        self.kernels.enqueue("recurrent_gated_delta_rule_f16",
                            [self.B, self.H * v_blocks, 16],
                            [1, 1, 16],
                            q,
                            k,
                            v,
                            g,
                            beta,
                            hidden_states,
                            o
                            )

if __name__ == "__main__":
    cl.profiling(True)
    # batch_size, max_kv_len = 16, 1024 
    # qkv=[16, 1024, 1152] float16  position_id_base=0
    B = 1
    H  = 16
    T = 32
    K = 128
    V = 128
    torch.set_printoptions(sci_mode=False)
    # torch.manual_seed(42)
    dtype = torch.float32
    q = torch.randn(B*H*T*K, dtype=dtype).reshape([B, H, T, K])
    k = torch.randn(B*H*T*K, dtype=dtype).reshape([B, H, T, K])
    v = torch.randn(B*H*T*V, dtype=dtype).reshape([B, H, T, V])
    beta = torch.rand(B, H, T, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, H, T, dtype=dtype))
    o = torch.randn(B, H, T, V, dtype=dtype).to(v)
    # hidden_states = torch.randn(B, H, K, V, dtype=dtype)

    # q = torch.zeros(B*H*T*K, dtype=dtype).reshape([B, H, T, K])
    # k = torch.zeros(B*H*T*K, dtype=dtype).reshape([B, H, T, K])
    # v = torch.zeros(B*H*T*V, dtype=dtype).reshape([B, H, T, V])
    # beta = torch.zeros(B*H*T, dtype=dtype).reshape([B, H, T])
    # g = torch.zeros(B*H*T, dtype=dtype).reshape([B, H, T])
    # o = torch.zeros(B, H, T, V).to(v)
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    hidden_states = torch.randn(B, H, V, K, dtype=dtype)
    # for h in range(H):
    #     for t in range(T):
    #         q[0, h, t, :] = torch.arange(t + h + 1, t + h + 1 + K)
    #         k[0, h, t, :] = torch.arange(t + h + 2, t + h + 2 + K)
    #         v[0, h, t, :] = torch.arange(t + h + 3, t + h + 3 + V)
    #         beta[0, h, t] = 0.25
    #         g[0, h, t] = 0.24
    #         hidden_states[0, h, :, :] = torch.ones([K, V])
    cl_q = to_cl(q)
    cl_k = to_cl(k)
    cl_v = to_cl(v)
    cl_g = to_cl(g)
    cl_beta = to_cl(beta)
    cl_o = to_cl(o)
    recurrnGDN = RecurrentGDN(B, H, T, K, V)
    n_times = 5
    print(n_times)
    for i in range(n_times):
        cl_hidden_states = to_cl(hidden_states)
        recurrnGDN(cl_q, cl_k, cl_v, cl_g, cl_beta, cl_hidden_states, cl_o)
    durs = cl.finish()
    Bsize = (q.numel() + k.numel() + v.numel() + g.numel() + beta.numel() + o.numel()) * 4
    for ns in durs:
        print(f"{Bsize*1e-6:.3f} MB {ns*1e-6:.3f} ms, BW: { Bsize/ns : .2f} GB/s")
    o_ref, h_ref = recurrent_gated_delta_rule_ref(q, k, v, beta, g, 1.0, hidden_states, output_final_state=True)
    # print("o ", to_torch(cl_o))
    # print("o_fef", o_ref)
    print(assert_close("11111", o_ref, to_torch(cl_o),0.002, 1e-6))
    print(assert_close("h_ref", h_ref, to_torch(cl_hidden_states), 0.002, 1e-6))

    print("\n=== fp16 path ===")
    q_h = q.half()
    k_h = k.half()
    v_h = v.half()
    beta_h = beta.half()
    g_h = g.half()
    o_h = o.half()
    hidden_states_h = hidden_states.half()

    cl_q_h = to_cl(q_h)
    cl_k_h = to_cl(k_h)
    cl_v_h = to_cl(v_h)
    cl_g_h = to_cl(g_h)
    cl_beta_h = to_cl(beta_h)
    cl_o_h = to_cl(o_h)
    recurrnGDN_h = RecurrentGDNHalf(B, H, T, K, V)
    for i in range(n_times):
        cl_hidden_states_h = to_cl(hidden_states_h)
        recurrnGDN_h(cl_q_h, cl_k_h, cl_v_h, cl_g_h, cl_beta_h, cl_hidden_states_h, cl_o_h)
    cl.finish()
    Bsize = (q.numel() + k.numel() + v.numel() + g.numel() + beta.numel() + o.numel()) * 4
    for ns in durs:
        print(f"{Bsize*1e-6:.3f} MB {ns*1e-6:.3f} ms, BW: { Bsize/ns : .2f} GB/s")
    o_ref_h, h_ref_h = recurrent_gated_delta_rule_ref(q_h, k_h, v_h, beta_h, g_h, 1.0, hidden_states_h, output_final_state=True)
    print(assert_close("fp16", o_ref_h, to_torch(cl_o_h).float(), 0.01, 1e-3))
    print(assert_close("fp16_h", h_ref_h, to_torch(cl_hidden_states_h).float(), 0.01, 1e-3))
