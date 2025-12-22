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

__kernel void recurrent_gated_delta_rule(__global half * q, 
    __global half * k, 
    __global half * v, 
    __global half * g,
    __global half * beta,
    __global half * initial_state) {
    int b = get_global_id(0);
    int h = get_global_id(1);
    int i_v = get_global_id(2);
    int BATCH_STRIDE = BATCH_NUM * K_HEAD_NUMS * SEQ_LEN;
    int HEAD_STRIDE = SEQ_LEN * K_HEAD_DIMS;
    half* q_ptr = q + b * BATCH_STRIDE + h * HEAD_STRIDE;
    half* k_ptr = k + b * BATCH_STRIDE + h * HEAD_STRIDE;
    half* v_ptr = v + b * BATCH_STRIDE + h * HEAD_STRIDE;
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
        printf("b_g %f\n", b_g);
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
        float output  = dot_product(init_state, b_q, K_HEAD_DIMS);
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
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        h = h.clone()
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)
    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h

class RecurrentGDN:
    def __init__(self, B, H, T, K, V):
        self.B = B
        self.H = H
        self.T = T
        self.K = K
        self.V = V
        self.kernels = kernel_cache(cl_kernel_sources, f"-DBATCH_NUM={self.B} -DK_HEAD_NUMS={self.H} -DSEQ_LEN={self.T} -DK_HEAD_DIMS={self.K}", "./dump/")
    
    def __call__(self, q, k, v, g, beta, hidden_states):
        self.kernels.enqueue("recurrent_gated_delta_rule",
                            [self.B, self.H, self.V],
                            [1, 1, 1],
                            q,
                            k,
                            v,
                            g,
                            beta,
                            hidden_states
                            )

if __name__ == "__main__":
    cl.profiling(True)
    # batch_size, max_kv_len = 16, 1024 
    # qkv=[16, 1024, 1152] float16  position_id_base=0
    B = 1
    H  = 1
    T = 2
    K = 16
    V = 16
    torch.set_printoptions(sci_mode=False)
    q = torch.zeros(B*H*T*K, dtype=torch.float16).reshape([B, H, T, K])
    k = torch.zeros(B*H*T*K, dtype=torch.float16).reshape([B, H, T, K])
    v = torch.zeros(B*H*T*V, dtype=torch.float16).reshape([B, H, T, V])
    beta = torch.zeros(B*H*T, dtype=torch.float16).reshape([B, H, T])
    g = torch.zeros(B*H*T, dtype=torch.float16).reshape([B, H, T])
    hidden_states = torch.randn(B, H, K, V, dtype=torch.float16)
    for h in range(H):
        for t in range(T):
            q[0, h, t, :] = 0 + h + 1
            k[0, h, t, :] = 0 + h + 2
            v[0, h, t, :] = 0 + h + 3
            beta[0, h, t] = 1
            g[0, h, t] = 0
            # hidden_states[0, h, :, :] = torch.ones([K, V])


    print("q ", q)
    print("k ", k)
    print("v ", v)
    cl_q = to_cl(q)
    cl_k = to_cl(k)
    cl_v = to_cl(v)
    cl_g = to_cl(g)
    cl_beta = to_cl(beta)
    cl_hidden_states = to_cl(hidden_states)
    recurrnGDN = RecurrentGDN(B, H, T, K, V)
    recurrnGDN(cl_q, cl_k, cl_v, cl_g, cl_beta, cl_hidden_states)
    o, h = recurrent_gated_delta_rule_ref(q, k, v, beta, g, 1.0, hidden_states, output_final_state=True)
    print("o ", o)
    print("h ", h)
    # print(to_torch(cl_q))
    # for i in range(10):
    #     vecAdd(input_a, input_b, output)

    # durs = cl.finish()
    # Bsize = (input_a.numel + input_b.numel + output.numel) * 2
    # for ns in durs:
    #     print(f"{Bsize*1e-6:.3f} MB {ns*1e-6:.3f} ms, BW: { Bsize/ns : .2f} GB/s")
