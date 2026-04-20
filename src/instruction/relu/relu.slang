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
}

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
    uint idx = threadId.x;
    if (idx >= pc.total)
        return;

    DTYPE val = a[idx];
    b[idx] = max((DTYPE)0, val);
}
