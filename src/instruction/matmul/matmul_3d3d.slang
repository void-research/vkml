struct PushConstants {
    uint batch;
    uint m;
    uint k;
    uint n;
    uint stride_a0;
    uint stride_a1;
    uint stride_a2;
    uint stride_b0;
    uint stride_b1;
    uint stride_b2;
    uint stride_c0;
    uint stride_c1;
    uint stride_c2;
};

[[vk::push_constant]]
PushConstants dims;

[[vk::binding(0)]]
RWStructuredBuffer<DTYPE> a;

[[vk::binding(1)]]
RWStructuredBuffer<DTYPE> b;

[[vk::binding(2)]]
RWStructuredBuffer<DTYPE> c;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main(uint3 threadId: SV_DispatchThreadID) {
    uint col = threadId.x;
    uint row = threadId.y;
    uint batch = threadId.z;

    if (row >= dims.m || col >= dims.n || batch >= dims.batch) {
        return;
    }

    DTYPE sum = DTYPE(0.0);

    for (uint i = 0; i < dims.k; i++) {
        uint a_idx = batch * dims.stride_a0 + row * dims.stride_a1 + i * dims.stride_a2;
        uint b_idx = batch * dims.stride_b0 + i * dims.stride_b1 + col * dims.stride_b2;
        sum += a[a_idx] * b[b_idx];
    }

    uint c_idx = batch * dims.stride_c0 + row * dims.stride_c1 + col * dims.stride_c2;
    c[c_idx] = sum;
}
