struct PushConstants {
    uint batch;
    uint m;
    uint k;
    uint stride_a0;
    uint stride_a1;
    uint stride_a2;
    uint stride_b;
    uint stride_c0;
    uint stride_c1;
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
    uint row = threadId.x;
    uint batch = threadId.y;

    if (row >= dims.m || batch >= dims.batch) {
        return;
    }

    DTYPE sum = DTYPE(0.0);

    for (uint i = 0; i < dims.k; i++) {
        uint a_idx = batch * dims.stride_a0 + row * dims.stride_a1 + i * dims.stride_a2;
        uint b_idx = i * dims.stride_b;
        sum += a[a_idx] * b[b_idx];
    }

    uint c_idx = batch * dims.stride_c0 + row * dims.stride_c1;
    c[c_idx] = sum;
}
