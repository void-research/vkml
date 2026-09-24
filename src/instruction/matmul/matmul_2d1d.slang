[vk::constant_id(0)]
const int WORKGROUP_SIZE_X = 1;
[vk::constant_id(1)]
const int WORKGROUP_SIZE_Y = 1;
[vk::constant_id(2)]
const int WORKGROUP_SIZE_Z = 1;

struct PushConstants
{
    uint m;
    uint k;
    uint stride_a0;
    uint stride_a1;
    uint stride_b;
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
    uint row = groupId.x * WORKGROUP_SIZE_X + threadId.x;
    uint tid = threadId.y;

    T sum = T(0);

    if (row < pc.m)
    {
        for (uint i = tid; i < pc.k; i += WORKGROUP_SIZE_Y)
        {
            sum = sum + src1[row * pc.stride_a0 + i * pc.stride_a1] * src2[i * pc.stride_b];
        }
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

        if (tid == 0 && row < pc.m)
        {
            dst[row * pc.stride_c] = sdata[shmem_idx];
        }
    }
    else if (row < pc.m)
    {
        dst[row * pc.stride_c] = sum;
    }
}
