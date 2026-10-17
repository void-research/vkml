[vk::constant_id(0)]
const int WORKGROUP_SIZE_X = 1;
[vk::constant_id(1)]
const int WORKGROUP_SIZE_Y = 1;
[vk::constant_id(2)]
const int WORKGROUP_SIZE_Z = 1;

struct PushConstants<T> {
    ImmutablePtr<T> src;
    Ptr<T> dst;
    uint total;
    uint reduction_size;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main<T : IArithmetic>(
    uniform PushConstants<T> pc,
    uint3 threadId: SV_DispatchThreadID)
{
    uint gid = threadId.x;
    uint total = pc.total;
    if (gid >= total)
        return;

    uint base = gid * pc.reduction_size;
    T sum = T(0);
    for (uint i = 0u; i < pc.reduction_size; ++i)
    {
        sum = sum + pc.src[base + i];
    }

    pc.dst[gid] = sum / T(pc.reduction_size);
}
