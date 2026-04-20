#ifndef DTYPE
#define DTYPE float
#endif

[vk::constant_id(0)]
const int WORKGROUP_SIZE_X = 1;
[vk::constant_id(1)]
const int WORKGROUP_SIZE_Y = 1;
[vk::constant_id(2)]
const int WORKGROUP_SIZE_Z = 1;

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
    uint gid = threadId.x;
    c[gid] = a[gid] + b[gid];
}
