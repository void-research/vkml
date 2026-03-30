[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main(
    StructuredBuffer<DTYPE> a,
    StructuredBuffer<DTYPE> b,
    RWStructuredBuffer<DTYPE> c,
    uint3 threadId: SV_DispatchThreadID)
{
    uint gid = threadId.x;
    c[gid] = a[gid] + b[gid];
}
