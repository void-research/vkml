struct PC
{
    uint rank;
    uint pad;
    uint total;
    uint dims[8];
    uint strides_a[8];
    uint strides_b[8];
}

[[vk::push_constant]]
ConstantBuffer<PC> pc;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main(
    StructuredBuffer<DTYPE> a,
    StructuredBuffer<DTYPE> b,
    RWStructuredBuffer<DTYPE> c,
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
        idx /= d;
        offA += r * pc.strides_a[i];
        offB += r * pc.strides_b[i];
    }

    c[gid] = a[offA] - b[offB];
}
