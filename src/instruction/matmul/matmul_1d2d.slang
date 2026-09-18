[vk::constant_id(0)]
const int WORKGROUP_SIZE_X = 1;
[vk::constant_id(1)]
const int WORKGROUP_SIZE_Y = 1;
[vk::constant_id(2)]
const int WORKGROUP_SIZE_Z = 1;

struct PushConstants
{
    uint k;
    uint n;
    uint stride_a;
    uint stride_b0;
    uint stride_b1;
    uint stride_c;
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
    uint col = groupId.x * WORKGROUP_SIZE_X + threadId.x;
    uint tid = threadId.y;

    if (col >= pc.n) return;

    T sum = T(0);

    for (uint i = tid; i < pc.k; i += WORKGROUP_SIZE_Y)
    {
        sum = sum + src1[i * pc.stride_a] * src2[i * pc.stride_b0 + col * pc.stride_b1];
    }

    if (WORKGROUP_SIZE_Y > 1)
    {
        static groupshared T sdata[WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y];
        uint shmem_idx = threadId.x * WORKGROUP_SIZE_Y + tid;
        sdata[shmem_idx] = sum;
        GroupMemoryBarrierWithGroupSync();

        for (uint s = WORKGROUP_SIZE_Y / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                sdata[shmem_idx] = sdata[shmem_idx] + sdata[shmem_idx + s];
            }
            GroupMemoryBarrierWithGroupSync();
        }

        if (tid == 0)
        {
            dst[col * pc.stride_c] = sdata[shmem_idx];
        }
    }
    else
    {
        dst[col * pc.stride_c] = sum;
    }
}
