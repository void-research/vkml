#ifndef DTYPE
#define DTYPE float
#endif

[vk::constant_id(0)]
const int WORKGROUP_SIZE_X = 1;
[vk::constant_id(1)]
const int WORKGROUP_SIZE_Y = 1;
[vk::constant_id(2)]
const int WORKGROUP_SIZE_Z = 1;

struct PushConstants
{
    uint k;
    uint n;
    uint stride_a;
    uint stride_b0;
    uint stride_b1;
    uint stride_c;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
StructuredBuffer<DTYPE> a;

[[vk::binding(1)]]
StructuredBuffer<DTYPE> b;

[[vk::binding(2)]]
RWStructuredBuffer<DTYPE> c;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main(uint3 threadId: SV_DispatchThreadID)
{
    uint col = threadId.x;

    if (col >= pc.n)
    {
        return;
    }

    DTYPE sum = DTYPE(0.0);

    for (uint i = 0; i < pc.k; i++)
    {
        uint a_idx = i * pc.stride_a;
        uint b_idx = i * pc.stride_b0 + col * pc.stride_b1;
        sum += a[a_idx] * b[b_idx];
    }

    uint c_idx = col * pc.stride_c;
    c[c_idx] = sum;
}
