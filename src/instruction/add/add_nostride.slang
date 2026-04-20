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
StructuredBuffer<DTYPE> src1;
[[vk::binding(1)]]
StructuredBuffer<DTYPE> src2;
[[vk::binding(2)]]
RWStructuredBuffer<DTYPE> dst;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main(uint3 threadId: SV_DispatchThreadID)
{
    uint gid = threadId.x;
    dst[gid] = src1[gid] + src2[gid];
}
