struct PushConstants {
    uint m;
    uint k;
    uint n;
    uint stride_a0;
    uint stride_a1;
    uint stride_b0;
    uint stride_b1;
    uint stride_y0;
    uint stride_y1;
    uint trans_a;
    uint trans_b;
    float alpha;
    float beta;
    uint has_c;
};

[[vk::push_constant]]
PushConstants params;

[[vk::binding(0)]]
RWStructuredBuffer<DTYPE> a;

[[vk::binding(1)]]
RWStructuredBuffer<DTYPE> b;

[[vk::binding(2)]]
RWStructuredBuffer<DTYPE> c;

[[vk::binding(3)]]
RWStructuredBuffer<DTYPE> y;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main(uint3 threadId: SV_DispatchThreadID) {
    uint row = threadId.y;
    uint col = threadId.x;

    if (row >= params.m || col >= params.n) {
        return;
    }

    DTYPE sum = DTYPE(0.0);

    uint stride_a_k = (params.trans_a != 0) ? params.stride_a0 : params.stride_a1;
    uint stride_a_m = (params.trans_a != 0) ? params.stride_a1 : params.stride_a0;
    uint stride_b_k = (params.trans_b != 0) ? params.stride_b1 : params.stride_b0;
    uint stride_b_n = (params.trans_b != 0) ? params.stride_b0 : params.stride_b1;

    for (uint i = 0; i < params.k; i++) {
        uint a_idx = row * stride_a_m + i * stride_a_k;
        uint b_idx = i * stride_b_k + col * stride_b_n;
        sum += a[a_idx] * b[b_idx];
    }

    DTYPE result = DTYPE(params.alpha) * sum;

    if (params.has_c != 0) {
        uint c_idx = row * params.n + col;
        result += DTYPE(params.beta) * c[c_idx];
    }

    uint y_idx = row * params.stride_y0 + col * params.stride_y1;
    y[y_idx] = result;
}
