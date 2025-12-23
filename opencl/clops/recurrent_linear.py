cl_kernel_sources = r'''
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

__kernel void recurrent_gated_delta_rule(__global float * q, 
    __global float * k, 
    __global float * v, 
    __global float * g,
    __global float * beta,
    __global float * initial_state,
    __global float * output) {
    int b = get_global_id(0);
    int h = get_global_id(1);
    int i_v = get_group_id(2);
    int BATCH_STRIDE = BATCH_NUM * K_HEAD_NUMS * SEQ_LEN;
    int HEAD_STRIDE = SEQ_LEN * K_HEAD_DIMS;
    float* q_ptr = q + b * BATCH_STRIDE + h * HEAD_STRIDE;
    float* k_ptr = k + b * BATCH_STRIDE + h * HEAD_STRIDE;
    float* v_ptr = v + b * BATCH_STRIDE + h * HEAD_STRIDE;
    float init_state[K_HEAD_DIMS] = {0};
    float b_k[K_HEAD_DIMS] = {0};
    float b_q[K_HEAD_DIMS] = {0};
    for (int j = 0; j < K_HEAD_DIMS; j++) {
        init_state[j] = initial_state[b * K_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS + h * K_HEAD_DIMS * K_HEAD_DIMS + j * K_HEAD_DIMS + i_v];
    }
    for (int i = 0; i < SEQ_LEN; i++) {
        float b_g = g[b * K_HEAD_NUMS * SEQ_LEN + h * SEQ_LEN + i];
        float b_beta = beta[b * K_HEAD_NUMS * SEQ_LEN + h * SEQ_LEN + i];
        b_g = exp(b_g);
        for (int j = 0; j < K_HEAD_DIMS; j++) {
            b_k[j] = k_ptr[i * K_HEAD_DIMS  + j];
            b_q[j] = q_ptr[i * K_HEAD_DIMS  + j];
        }
        // h0 * g
        scale(init_state, b_g, K_HEAD_DIMS);
        float h_k = dot_product(init_state, b_k, K_HEAD_DIMS);
        float b_v = v_ptr[i_v + i * K_HEAD_DIMS];
        b_v -= h_k;
        // b_v * b_k
        b_v *= b_beta;
        scale(b_k, b_v, K_HEAD_DIMS);
        // h = h0 + update
        add(init_state, b_k, K_HEAD_DIMS);
        float b_output  = dot_product(init_state, b_q, K_HEAD_DIMS);
        output[b * K_HEAD_NUMS * SEQ_LEN * K_HEAD_DIMS + h * SEQ_LEN * K_HEAD_DIMS + i * K_HEAD_DIMS + i_v] = b_output;
        //printf("b %d h %d i_v %d seq %i output %f init_state %f\n", b, h, i_v, i, b_output, init_state[0]);
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

'''

cm_kernel_sources = r'''
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
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void recurrent_gated_delta_rule(__global float * q, 
    __global float * k, 
    __global float * v, 
    __global float * g,
    __global float * beta,
    __global float * initial_state,
    __global float * output) {
    int b = get_global_id(0);
    int h = get_global_id(1);
    int i_v = get_group_id(2);
    int id_local = get_local_id(2);
    int BATCH_STRIDE = BATCH_NUM * K_HEAD_NUMS * SEQ_LEN;
    int HEAD_STRIDE = SEQ_LEN * K_HEAD_DIMS;
    float* q_ptr = q + b * BATCH_STRIDE + h * HEAD_STRIDE;
    float* k_ptr = k + b * BATCH_STRIDE + h * HEAD_STRIDE;
    float* v_ptr = v + b * BATCH_STRIDE + h * HEAD_STRIDE;
    float init_state[K_HEAD_DIMS] = {0};
    float b_k[K_HEAD_DIMS] = {0};
    float b_q[K_HEAD_DIMS] = {0};
    int id_sg = get_sub_group_id();
    __local float variance[1];
    int id_sg_local = get_sub_group_local_id();
    for (int j = id_local; j < K_HEAD_DIMS; j+= 16) {
        init_state[j] = initial_state[b * K_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS + h * K_HEAD_DIMS * K_HEAD_DIMS + j * K_HEAD_DIMS + i_v];
    }
    for (int i = 0; i < SEQ_LEN; i++) {
        float b_g = g[b * K_HEAD_NUMS * SEQ_LEN + h * SEQ_LEN + i];
        float b_beta = beta[b * K_HEAD_NUMS * SEQ_LEN + h * SEQ_LEN + i];
        b_g = exp(b_g);
        for (int j = id_local; j < K_HEAD_DIMS; j+= 16) {
            b_k[j] = k_ptr[i * K_HEAD_DIMS  + j];
            b_q[j] = q_ptr[i * K_HEAD_DIMS  + j];
        }
        // h0 * g
        for (size_t n = id_local; n < K_HEAD_DIMS; n+= 16) {
            init_state[n] *= b_g;
        }
        float hk_result = 0.0f;
        for (size_t n = id_local; n < K_HEAD_DIMS; n+= 16) {
            hk_result = fma(init_state[n], b_k[n], hk_result);
        }
        hk_result = sub_group_reduce_add(hk_result);
        if (id_sg_local == 0)
            variance[id_sg] = hk_result;
        barrier(CLK_LOCAL_MEM_FENCE);
        hk_result = variance[id_sg];   

        float b_v = v_ptr[i_v + i * K_HEAD_DIMS];
        b_v -= hk_result;
        // b_v * b_k
        b_v *= b_beta;
        // h0 = h0 + b_k * b_v;
        for (size_t n = id_local; n < K_HEAD_DIMS; n+= 16) {
            init_state[n] = fma(b_k[n], b_v, init_state[n]);
        };
        // float b_output  = dot_product(init_state, b_q, K_HEAD_DIMS);
        float b_output = 0.0f;
        for (size_t n = id_local; n < K_HEAD_DIMS; n+= 16) {
            b_output = fma(init_state[n], b_q[n], b_output);
        }
        b_output = sub_group_reduce_add(b_output);
        if (id_sg_local == 0) {
            variance[id_sg] = b_output;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id_local == 0) {
            output[b * K_HEAD_NUMS * SEQ_LEN * K_HEAD_DIMS + h * SEQ_LEN * K_HEAD_DIMS + i * K_HEAD_DIMS + i_v] = b_output;
            // printf("b %d h %d i_v %d seq %i output %f init_state %f\n", b, h, i_v, i, b_output, init_state[0]);    
        }
        //
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


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return "Good Result"
    else:
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
        h = initial_state
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
        self.kernels.enqueue("recurrent_gated_delta_rule",
                            [self.B, self.H, self.V * 16],
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
    hidden_states = torch.randn(B, H, K, V, dtype=dtype)
    # for h in range(H):
    #     for t in range(T):
    #         q[0, h, t, :] = torch.arange(t + h + 1, t + h + 1 + K)
    #         k[0, h, t, :] = torch.arange(t + h + 2, t + h + 2 + K)
    #         v[0, h, t, :] = torch.arange(t + h + 3, t + h + 3 + V)
    #         beta[0, h, t] = 0.25
    #         g[0, h, t] = 0.24
    #         hidden_states[0, h, :, :] = torch.ones([K, V])


    print("q ", q)
    print("k ", k)
    print("v ", v)
    cl_q = to_cl(q)
    cl_k = to_cl(k)
    cl_v = to_cl(v)
    cl_g = to_cl(g)
    cl_beta = to_cl(beta)
    cl_hidden_states = to_cl(hidden_states)
    cl_o = to_cl(o)
    recurrnGDN = RecurrentGDN(B, H, T, K, V)
    n_times = 1
    print(n_times)
    for i in range(n_times):
        recurrnGDN(cl_q, cl_k, cl_v, cl_g, cl_beta, cl_hidden_states, cl_o)
    durs = cl.finish()
    Bsize = (q.numel() + k.numel() + v.numel() + g.numel() + beta.numel() + o.numel()) * 4
    for ns in durs:
        print(f"{Bsize*1e-6:.3f} MB {ns*1e-6:.3f} ms, BW: { Bsize/ns : .2f} GB/s")
    o_ref, h_ref = recurrent_gated_delta_rule_ref(q, k, v, beta, g, 1.0, hidden_states, output_final_state=True)
    # print("o ", to_torch(cl_o))
    # print("o_fef", o_ref)
    print(assert_close("11111", o_ref, to_torch(cl_o),0.002))
