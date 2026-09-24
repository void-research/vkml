[vk::constant_id(0)]
const int WORKGROUP_SIZE_X = 1;
[vk::constant_id(1)]
const int WORKGROUP_SIZE_Y = 1;
[vk::constant_id(2)]
const int WORKGROUP_SIZE_Z = 1;

struct PushConstants
{
    uint batch;
    uint k;
    uint n;
    uint total; // batch * n
    uint stride_a;
    uint stride_b0;
    uint stride_b1;
    uint stride_b2;
    uint stride_c0;
    uint stride_c1;
}

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main<T : IArithmetic>(
    StructuredBuffer<T> src1,
    StructuredBuffer<T> src2,
    RWStructuredBuffer<T> dst,
    uint3 groupId: SV_GroupID,
    uint3 threadId: SV_GroupThreadID)
{
    uint flat = groupId.x * WORKGROUP_SIZE_X + threadId.x;

    if (flat >= pc.total)
        return;

    uint batch_idx = flat / pc.n;
    uint col = flat % pc.n;

    T sum = T(0);

    for (uint i = 0; i < pc.k; i++)
    {
        uint a_idx = i * pc.stride_a;
        uint b_idx = batch_idx * pc.stride_b0 + i * pc.stride_b1 + col * pc.stride_b2;
        sum = sum + src1[a_idx] * src2[b_idx];
    }

    uint c_idx = batch_idx * pc.stride_c0 + col * pc.stride_c1;
    dst[c_idx] = sum;
}
