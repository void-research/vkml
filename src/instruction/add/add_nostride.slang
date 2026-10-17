[vk::constant_id(0)]
const int WORKGROUP_SIZE_X = 1;
[vk::constant_id(1)]
const int WORKGROUP_SIZE_Y = 1;
[vk::constant_id(2)]
const int WORKGROUP_SIZE_Z = 1;

struct PushConstants<T> {
    ImmutablePtr<T> src1;
    ImmutablePtr<T> src2;
    Ptr<T> dst;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main<T : IArithmetic>(
    uniform PushConstants<T> pc,
    uint3 threadId: SV_DispatchThreadID)
{
    uint gid = threadId.x;
    pc.dst[gid] = pc.src1[gid] + pc.src2[gid];
}
