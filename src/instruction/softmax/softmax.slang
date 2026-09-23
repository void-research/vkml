struct PushConstants
{
    uint batch_size;
    uint feature_size;
}

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main<T : __BuiltinFloatingPointType>(
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

    const uint WG_SIZE = 256;
    static groupshared T max_vals[WG_SIZE];
    static groupshared T sum_vals[WG_SIZE];

    // Calculate base index for this batch
    uint base_idx = batch_idx * pc.feature_size;

    T local_max = T(0.0);
    T local_sum = T(0.0);

    // Pass 1: Compute local max and sum simultaneously (Online Softmax)
    if (local_id < pc.feature_size)
    {
        local_max = src[base_idx + local_id];
        local_sum = T(1.0);

        for (uint i = local_id + WG_SIZE; i < pc.feature_size; i += WG_SIZE)
        {
            T val = src[base_idx + i];
            if (val > local_max)
            {
                local_sum = local_sum * exp(local_max - val) + T(1.0);
                local_max = val;
            }
            else
            {
                local_sum += exp(val - local_max);
            }
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
            T m1 = max_vals[local_id];
            T s1 = sum_vals[local_id];
            T m2 = max_vals[local_id + stride];
            T s2 = sum_vals[local_id + stride];

            if (s1 == T(0.0))
            {
                max_vals[local_id] = m2;
                sum_vals[local_id] = s2;
            }
            else if (s2 == T(0.0))
            {
                max_vals[local_id] = m1;
                sum_vals[local_id] = s1;
            }
            else
            {
                T m_new = max(m1, m2);
                T s_new = s1 * exp(m1 - m_new) + s2 * exp(m2 - m_new);
                max_vals[local_id] = m_new;
                sum_vals[local_id] = s_new;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    T final_max = max_vals[0];
    T final_sum = sum_vals[0];

    // Pass 2: Compute final normalized value
    for (uint i = local_id; i < pc.feature_size; i += WG_SIZE)
    {
        uint idx = base_idx + i;
        dst[idx] = exp(src[idx] - final_max) / final_sum;
    }
}

