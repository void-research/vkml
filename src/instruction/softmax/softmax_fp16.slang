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

    // Calculate base index for this batch
    uint base_idx = batch_idx * pc.feature_size;

    half local_max = half(-65504.0); // -HALF_MAX
    half local_sum = half(0.0);

    // Pass 1: Compute local max and sum simultaneously (Online Softmax)
    for (uint i = local_id; i < pc.feature_size; i += WG_SIZE)
    {
        half val = src[base_idx + i];
        if (val > local_max)
        {
            local_sum = local_sum * exp(local_max - val) + half(1.0);
            local_max = val;
        }
        else
        {
            local_sum += exp(val - local_max);
        }
    }

    max_vals[local_id] = local_max;
    sum_vals[local_id] = local_sum;

    // Reduction to find the global maximum and sum
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = WG_SIZE / 2; stride > 0; stride = stride / 2)
    {
        if (local_id < stride)
        {
            half m1 = max_vals[local_id];
            half s1 = sum_vals[local_id];
            half m2 = max_vals[local_id + stride];
            half s2 = sum_vals[local_id + stride];

            half m_new = max(m1, m2);
            half s_new = s1 * exp(m1 - m_new) + s2 * exp(m2 - m_new);

            max_vals[local_id] = m_new;
            sum_vals[local_id] = s_new;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    half final_max = max_vals[0];
    half final_sum = sum_vals[0];

    // Pass 2: Compute final normalized value
    for (uint i = local_id; i < pc.feature_size; i += WG_SIZE)
    {
        uint idx = base_idx + i;
        dst[idx] = exp(src[idx] - final_max) / final_sum;
    }
}
