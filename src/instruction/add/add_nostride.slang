[vk::constant_id(0)]
const int WORKGROUP_SIZE_X = 1;
[vk::constant_id(1)]
const int WORKGROUP_SIZE_Y = 1;
[vk::constant_id(2)]
const int WORKGROUP_SIZE_Z = 1;

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
