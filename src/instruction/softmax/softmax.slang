[vk::constant_id(0)]
const int WORKGROUP_SIZE_X = 1;
[vk::constant_id(1)]
const int WORKGROUP_SIZE_Y = 1;
[vk::constant_id(2)]
const int WORKGROUP_SIZE_Z = 1;

struct PushConstants
{
    uint batch_size;
    uint feature_size;
}

[[vk::push_constant]]
PushConstants pc;

// TODO: FIX, use slang exp() function somehow
T exp_t<T : IFloat>(T x)
{
    T clamped = max(T(-10), min(T(10), x));

    // Approximate e^x as (1 + x / 256)^256 via repeated squaring.
    T y = T(1) + clamped * (T(1) / T(256));
    y = y * y;
    y = y * y;
    y = y * y;
    y = y * y;
    y = y * y;
    y = y * y;
    y = y * y;
    y = y * y;

    return y;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, 1, 1)]
void main<T : IFloat>(
    StructuredBuffer<T> src,
    RWStructuredBuffer<T> dst,
    uint3 threadId: SV_GroupThreadID, uint3 groupId: SV_GroupID)
{
    uint local_id = threadId.x;
    uint batch_idx = groupId.x;

    if (batch_idx >= pc.batch_size)
    {
        return;
    }

    static groupshared T max_vals[WORKGROUP_SIZE_X];
    static groupshared T sum_vals[WORKGROUP_SIZE_X];

    // Initialize shared memory
    max_vals[local_id] = T(-3.402823466e+38); // -FLT_MAX
    sum_vals[local_id] = T(0);

    // Calculate base index for this batch
    uint base_idx = batch_idx * pc.feature_size;

    // Find maximum in this batch (for numerical stability)
    for (uint i = local_id; i < pc.feature_size; i = i + WORKGROUP_SIZE_X)
    {
        max_vals[local_id] = max(max_vals[local_id], src[base_idx + i]);
    }

    // Reduction to find the maximum
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = WORKGROUP_SIZE_X / 2; stride > 0; stride = stride / 2)
    {
        if (local_id < stride)
        {
            max_vals[local_id] = max(max_vals[local_id], max_vals[local_id + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    T max_val = max_vals[0];

    // Compute exponentials and partial sums
    for (uint i = local_id; i < pc.feature_size; i = i + WORKGROUP_SIZE_X)
    {
        uint idx = base_idx + i;
        T exp_val = exp_t(src[idx] - max_val);
        dst[idx] = exp_val;
        sum_vals[local_id] = sum_vals[local_id] + exp_val;
    }

    GroupMemoryBarrierWithGroupSync();
    for (uint stride = WORKGROUP_SIZE_X / 2; stride > 0; stride = stride / 2)
    {
        if (local_id < stride)
        {
            sum_vals[local_id] = sum_vals[local_id] + sum_vals[local_id + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    T sum = sum_vals[0];

    // Normalize by sum
    for (uint i = local_id; i < pc.feature_size; i = i + WORKGROUP_SIZE_X)
    {
        uint idx = base_idx + i;
        dst[idx] = dst[idx] / sum;
    }
}
