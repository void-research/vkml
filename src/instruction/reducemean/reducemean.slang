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
    uint total;
    uint reduction_size;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
StructuredBuffer<DTYPE> a;

[[vk::binding(1)]]
RWStructuredBuffer<DTYPE> b;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main(uint3 threadId: SV_DispatchThreadID)
{
    uint gid = threadId.x;
    uint total = pc.total;
    if (gid >= total)
        return;

    uint base = gid * pc.reduction_size;
    DTYPE sum = DTYPE(0.0);
    for (uint i = 0u; i < pc.reduction_size; ++i)
    {
        sum += a[base + i];
    }

    b[gid] = sum / DTYPE(pc.reduction_size);
}
