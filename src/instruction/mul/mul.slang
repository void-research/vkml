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
    uint rank;
    uint pad;
    uint total;
    uint dims[8];
    uint strides_a[8];
    uint strides_b[8];
}

[[vk::push_constant]]
PushConstants pc;

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
    uint total = pc.total;
    if (gid >= total)
        return;

    uint idx = gid;
    uint offA = 0;
    uint offB = 0;
    for (int i = int(pc.rank) - 1; i >= 0; --i)
    {
        uint d = pc.dims[i];
        uint r = idx % d;
        idx /= d;
        offA += r * pc.strides_a[i];
        offB += r * pc.strides_b[i];
    }

    dst[gid] = src1[offA] * src2[offB];
}
