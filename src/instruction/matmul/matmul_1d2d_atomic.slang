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
    uint num_slices;
    uint stride_a;
    uint stride_b0;
    uint stride_b1;
    uint stride_c;
};

[[vk::push_constant]]
PushConstants pc;

interface IAtomicAdd
{
    static void add(RWStructuredBuffer<This> buf, uint index, This val);
};

extension float : IAtomicAdd
{
    static void add(RWStructuredBuffer<float> buf, uint index, float val)
    {
        InterlockedAdd(buf[index], val);
    }
};

extension half : IAtomicAdd
{
    static void add(RWStructuredBuffer<half> buf, uint index, half val)
    {
        InterlockedAdd(buf[index], val);
    }
};

extension double : IAtomicAdd
{
    static void add(RWStructuredBuffer<double> buf, uint index, double val)
    {
        InterlockedAdd(buf[index], val);
    }
};

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main<T : IArithmetic & IAtomicAdd>(
    StructuredBuffer<T> src1,
    StructuredBuffer<T> src2,
    RWStructuredBuffer<T> dst,
    uint3 groupId: SV_GroupID,
    uint3 threadId: SV_GroupThreadID)
{
    uint col = groupId.x * WORKGROUP_SIZE_X + threadId.x;
    uint slice = groupId.y;
    uint tid = threadId.y;

    uint slice_size = (pc.k + pc.num_slices - 1) / pc.num_slices;
    uint k_start = slice * slice_size;
    uint k_end = min(k_start + slice_size, pc.k);

    T sum = T(0);

    if (col < pc.n && k_start < pc.k)
    {
        for (uint i = k_start + tid; i < k_end; i += WORKGROUP_SIZE_Y)
        {
            sum = sum + src1[i * pc.stride_a] * src2[i * pc.stride_b0 + col * pc.stride_b1];
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

        if (tid == 0 && col < pc.n)
        {
            T.add(dst, col * pc.stride_c, sdata[shmem_idx]);
        }
    }
    else if (col < pc.n)
    {
        T.add(dst, col * pc.stride_c, sum);
    }
}
