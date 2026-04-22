[vk::constant_id(0)]
const int WORKGROUP_SIZE_X = 1;
[vk::constant_id(1)]
const int WORKGROUP_SIZE_Y = 1;
[vk::constant_id(2)]
const int WORKGROUP_SIZE_Z = 1;

struct PushConstants
{
    uint batch;
    uint m;
    uint k;
    uint stride_a0;
    uint stride_a1;
    uint stride_a2;
    uint stride_b;
    uint stride_c0;
    uint stride_c1;
}

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main<T : IArithmetic>(
    StructuredBuffer<T> src1,
    StructuredBuffer<T> src2,
    RWStructuredBuffer<T> dst,
    uint3 threadId: SV_DispatchThreadID)
{
    uint row = threadId.x;
    uint batch = threadId.y;

    if (row >= pc.m || batch >= pc.batch)
    {
        return;
    }

    T sum = T(0);

    for (uint i = 0; i < pc.k; i++)
    {
        uint a_idx = batch * pc.stride_a0 + row * pc.stride_a1 + i * pc.stride_a2;
        uint b_idx = i * pc.stride_b;
        sum = sum + src1[a_idx] * src2[b_idx];
    }

    uint c_idx = batch * pc.stride_c0 + row * pc.stride_c1;
    dst[c_idx] = sum;
}
