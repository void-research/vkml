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
    uint rank;
    uint dims[8];
    uint strides_src[8];
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

    uint idx = gid;
    uint offSrc = 0;
    // Iterate from rank - 1 down to 0
    for (int i = int(pc.rank) - 1; i >= 0; --i)
    {
        uint d = pc.dims[i];
        uint r = idx % d;
        idx = idx / d;
        offSrc = offSrc + r * pc.strides_src[i];
    }

    pc.dst[gid] = pc.src[offSrc];
}
