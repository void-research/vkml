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
    uint total;
    uint rank;
    uint dims[8];
    uint strides_a[8];
    uint strides_b[8];
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
    uint offA = 0;
    uint offB = 0;
    for (int i = int(pc.rank) - 1; i >= 0; --i)
    {
        uint d = pc.dims[i];
        uint r = idx % d;
        idx = idx / d;
        offA = offA + r * pc.strides_a[i];
        offB = offB + r * pc.strides_b[i];
    }

    pc.dst[gid] = pc.src1[offA] - pc.src2[offB];
}
