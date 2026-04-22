struct PushConstants
{
    uint batch_size;
    uint feature_size;
}

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
    StructuredBuffer<half> src,
    RWStructuredBuffer<half> dst,
    uint3 threadId: SV_GroupThreadID, uint3 groupId: SV_GroupID)
{
    uint local_id = threadId.x;
    uint batch_idx = groupId.x;

    if (batch_idx >= pc.batch_size)
    {
        return;
    }

    const uint WG_SIZE = 256;
    static groupshared half max_vals[WG_SIZE];
    static groupshared half sum_vals[WG_SIZE];

    // Initialize shared memory
    max_vals[local_id] = half(-65504.0); // -HALF_MAX
    sum_vals[local_id] = half(0.0);

    // Calculate base index for this batch
    uint base_idx = batch_idx * pc.feature_size;

    // Find maximum in this batch (for numerical stability)
    for (uint i = local_id; i < pc.feature_size; i = i + WG_SIZE)
    {
        max_vals[local_id] = max(max_vals[local_id], src[base_idx + i]);
    }

    // Reduction to find the maximum
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = WG_SIZE / 2; stride > 0; stride = stride / 2)
    {
        if (local_id < stride)
        {
            max_vals[local_id] = max(max_vals[local_id], max_vals[local_id + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    half max_val = max_vals[0];

    // Compute exponentials and partial sums
    for (uint i = local_id; i < pc.feature_size; i = i + WG_SIZE)
    {
        uint idx = base_idx + i;
        half exp_val = exp(src[idx] - max_val);
        dst[idx] = exp_val;
        sum_vals[local_id] = sum_vals[local_id] + exp_val;
    }

    GroupMemoryBarrierWithGroupSync();
    for (uint stride = WG_SIZE / 2; stride > 0; stride = stride / 2)
    {
        if (local_id < stride)
        {
            sum_vals[local_id] = sum_vals[local_id] + sum_vals[local_id + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    half sum = sum_vals[0];

    // Normalize by sum
    for (uint i = local_id; i < pc.feature_size; i = i + WG_SIZE)
    {
        uint idx = base_idx + i;
        dst[idx] = dst[idx] / sum;
    }
}
