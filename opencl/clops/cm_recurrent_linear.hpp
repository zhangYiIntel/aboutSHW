//# CM kernel for flash attn, reference
#include <cm/cm.h>
#include <cm/cmtl.h>
// #include <math.h>

//# CM-compiler is C++17
static_assert(__cplusplus >= 201703L);

extern "C" _GENX_MAIN_ void vadd(SurfaceIndex ibuf0 [[type("buffer_t")]],
                                 SurfaceIndex ibuf1 [[type("buffer_t")]],
                                 SurfaceIndex obuf [[type("buffer_t")]]) {
    unsigned tid = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    printf("tid %d\n", tid);
    vector<half, 32> in0;
    read(ibuf0, tid * 32 * sizeof(half), in0);

    vector<half, 32> in1;
    read(ibuf1, tid * 32 * sizeof(half), in1);

    in0 += in1;

    write(obuf, tid * 32 * sizeof(half), in0);
}

template <int k_num_heads, int v_num_heads, int k_head_dims, int v_head_dims>
void recurrent_linear_attn(int b_idx,
                           int head_idx,
                           int head_dim_t_idx,
                           SurfaceIndex q [[type("buffer_t")]],
                           SurfaceIndex k [[type("buffer_t")]],
                           SurfaceIndex v [[type("buffer_t")]],
                           SurfaceIndex g [[type("buffer_t")]],
                           SurfaceIndex beta [[type("buffer_t")]],
                           SurfaceIndex initial_state [[type("buffer_t")]],
                           SurfaceIndex output [[type("buffer_t")]]) {
    constexpr int v_head_dim_per_t = 16;  // 16
    vector<float, v_head_dim_per_t * k_head_dims> h0;
// h0 [B, H, V, K]
#pragma unroll
    for (int i = 0; i < v_head_dim_per_t; i++) {
        int v_head_dim_idx = head_dim_t_idx * v_head_dim_per_t + i;
        int stride = b_idx * k_num_heads * v_head_dims * k_head_dims + head_idx * v_head_dims * k_head_dims +
                     v_head_dim_idx * k_head_dims;
        if constexpr (k_head_dims == 128) {
            h0.select<64, 1>(k_head_dims * i) = cm_load<float, 64>(initial_state, stride * 4);
            h0.select<64, 1>(k_head_dims * i + 64) = cm_load<float, 64>(initial_state, stride + 64 * 4);
        } else if constexpr (k_head_dims <= 64) {
            h0.select<k_head_dims, 1>(k_head_dims * i) = cm_load<float, k_head_dims>(initial_state, stride * 4);
        }
    }

    for (int s = 0; s < SEQ_LEN; s++) {
        // beta B, T, HV
        // g B, T, HV
        int stride = b_idx * SEQ_LEN * v_num_heads + s * v_num_heads + head_idx;
        auto b_beta = cm_load<float, 1>(beta, stride * 4);
        auto b_g = cm_load<float, 1>(g, stride * 4);
        if (head_dim_t_idx == 0) {
            printf("b_idx %d head_idx %d head_dim_t_idx %d beta_cur %f b_g %f\n",
                   b_idx,
                   head_idx,
                   head_dim_t_idx,
                   b_beta[0],
                   b_g[0]);
        }
        // B, T, HK, K
        int qk_stride =
            b_idx * SEQ_LEN * k_num_heads * k_head_dims + s * k_num_heads * k_head_dims + head_idx * k_head_dims;

        vector<float, k_head_dims> b_q = cm_load<float, k_head_dims>(q, qk_stride * 4);
        vector<float, k_head_dims> b_k = cm_load<float, k_head_dims>(k, qk_stride * 4);
        // read_v
        // B, T, HV, V
        int v_stride = b_idx * SEQ_LEN * v_num_heads * v_head_dims + s * v_num_heads * v_head_dims +
                       head_idx * v_head_dims + head_dim_t_idx * v_head_dim_per_t;
        vector<float, v_head_dim_per_t> b_v = cm_load<float, v_head_dim_per_t>(v, v_stride * 4);
        // if (head_dim_t_idx == 0) {
        //     printf("b_idx %d head_idx %d head_dim_t_idx %d b_q %f b_q %f\n", b_idx, head_idx, head_dim_t_idx, b_q[0],
        //     b_q[1]); printf("b_idx %d head_idx %d head_dim_t_idx %d b_k %f b_k %f\n", b_idx, head_idx,
        //     head_dim_t_idx, b_k[0], b_k[1]);
        // }
        vector<float, v_head_dim_per_t> cur_output;
        vector<float, v_head_dim_per_t> h_k;
        constexpr float log2e = 1.4426950408889634f;
        float g_cur = cm_exp(b_g[0] * log2e);
#pragma unroll
        for (int i = 0; i < v_head_dim_per_t; i++) {
            h0.select<k_head_dims, 1>(k_head_dims * i) = h0.select<k_head_dims, 1>(k_head_dims * i) * g_cur;
            h_k[i] = cm_sum<float>(h0.select<k_head_dims, 1>(k_head_dims * i) * b_k);
            if (head_dim_t_idx == 0) {
                printf("b_idx %d head_idx %d head_dim_t_idx %d h_k %f g_cur %f\n",
                       b_idx,
                       head_idx,
                       head_dim_t_idx,
                       h_k[i],
                       g_cur);
            }
        }
        vector<float, v_head_dim_per_t> delta_full = (b_v - h_k) * b_beta[0];
        if (head_dim_t_idx == 0) {
            printf("b_idx %d head_idx %d head_dim_t_idx %d delta_full %f\n",
                   b_idx,
                   head_idx,
                   head_dim_t_idx,
                   delta_full[0]);
        }

#pragma unroll
        for (int i = 0; i < v_head_dim_per_t; i++) {
            float detla = delta_full[i];
            h0.select<k_head_dims, 1>(k_head_dims * i) = h0.select<k_head_dims, 1>(k_head_dims * i) + b_k * detla;
            cur_output[i] = cm_sum<float>(h0.select<k_head_dims, 1>(k_head_dims * i) * b_q);
            if (head_dim_t_idx == 0) {
                printf("b_idx %d head_idx %d head_dim_t_idx %d h_k %f\n",
                       b_idx,
                       head_idx,
                       head_dim_t_idx,
                       h0.select<k_head_dims, 1>(k_head_dims * i)[0]);
            }
        }
        // B, T, HV, V
        int output_stride = b_idx * SEQ_LEN * v_num_heads * v_head_dims + s * v_num_heads * v_head_dims +
                            head_idx * v_head_dims + head_dim_t_idx * v_head_dim_per_t;
        cm_store<float, v_head_dim_per_t>(output, output_stride * 4, cur_output);
    }
#pragma unroll
    for (int i = 0; i < v_head_dim_per_t; i++) {
        int v_head_dim_idx = head_dim_t_idx * v_head_dim_per_t + i;
        int stride = b_idx * k_num_heads * v_head_dims * k_head_dims + head_idx * v_head_dims * k_head_dims +
                     v_head_dim_idx * k_head_dims;
        cm_store<float, k_head_dims>(initial_state, stride * 4, h0.select<k_head_dims, 1>(k_head_dims * i));
    }
}

extern "C" _GENX_MAIN_ void recurrent_gated_delta_rule(SurfaceIndex q [[type("buffer_t")]],
                                                       SurfaceIndex k [[type("buffer_t")]],
                                                       SurfaceIndex v [[type("buffer_t")]],
                                                       SurfaceIndex g [[type("buffer_t")]],
                                                       SurfaceIndex beta [[type("buffer_t")]],
                                                       SurfaceIndex initial_state [[type("buffer_t")]],
                                                       SurfaceIndex output [[type("buffer_t")]]) {
    int b_idx = cm_group_id(0);
    int head_idx = cm_group_id(0);
    int head_dim_t_idx = cm_local_id(2);
    constexpr int k_num_heads = K_HEAD_NUMS;
    constexpr int v_num_heads = V_HEAD_NUMS;
    constexpr int k_head_dims = K_HEAD_DIMS;
    constexpr int v_head_dims = V_HEAD_DIMS;
    recurrent_linear_attn<k_num_heads, v_num_heads, k_head_dims, v_head_dims>(b_idx,
                                                                              head_idx,
                                                                              head_dim_t_idx,
                                                                              q,
                                                                              k,
                                                                              v,
                                                                              g,
                                                                              beta,
                                                                              initial_state,
                                                                              output);
}
