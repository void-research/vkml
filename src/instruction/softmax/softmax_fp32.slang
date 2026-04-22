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
    StructuredBuffer<float> src,
    RWStructuredBuffer<float> dst,
    uint3 threadId: SV_GroupThreadID, uint3 groupId: SV_GroupID)
{
    uint local_id = threadId.x;
    uint batch_idx = groupId.x;

    if (batch_idx >= pc.batch_size)
    {
        return;
    }

    const uint WG_SIZE = 256;
    static groupshared float max_vals[WG_SIZE];
    static groupshared float sum_vals[WG_SIZE];

    // Calculate base index for this batch
    uint base_idx = batch_idx * pc.feature_size;

    float local_max = -3.402823466e+38f; // -FLT_MAX
    float local_sum = 0.0f;

    // Pass 1: Compute local max and sum simultaneously (Online Softmax)
    for (uint i = local_id; i < pc.feature_size; i += WG_SIZE)
    {
        float val = src[base_idx + i];
        if (val > local_max)
        {
            local_sum = local_sum * exp(local_max - val) + 1.0f;
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
            float m1 = max_vals[local_id];
            float s1 = sum_vals[local_id];
            float m2 = max_vals[local_id + stride];
            float s2 = sum_vals[local_id + stride];

            float m_new = max(m1, m2);
            float s_new = s1 * exp(m1 - m_new) + s2 * exp(m2 - m_new);

            max_vals[local_id] = m_new;
            sum_vals[local_id] = s_new;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float final_max = max_vals[0];
    float final_sum = sum_vals[0];

    // Pass 2: Compute final normalized value
    for (uint i = local_id; i < pc.feature_size; i += WG_SIZE)
    {
        uint idx = base_idx + i;
        dst[idx] = exp(src[idx] - final_max) / final_sum;
    }
}
