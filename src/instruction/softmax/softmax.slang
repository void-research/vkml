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
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
RWStructuredBuffer<DTYPE> a;

[[vk::binding(1)]]
RWStructuredBuffer<DTYPE> b;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, 1, 1)]
void main(uint3 threadId: SV_GroupThreadID, uint3 groupId: SV_GroupID)
{
    uint local_id = threadId.x;
    uint batch_idx = groupId.x;

    if (batch_idx >= pc.batch_size)
    {
        return;
    }

    groupshared DTYPE max_vals[WORKGROUP_SIZE_X];
    groupshared DTYPE sum_vals[WORKGROUP_SIZE_X];

    // Initialize shared memory
    max_vals[local_id] = DTYPE(-3.402823466e+38); // -FLT_MAX
    sum_vals[local_id] = DTYPE(0.0);

    // Calculate base index for this batch
    uint base_idx = batch_idx * pc.feature_size;

    // Find maximum in this batch (for numerical stability)
    for (uint i = local_id; i < pc.feature_size; i += WORKGROUP_SIZE_X)
    {
        max_vals[local_id] = max(max_vals[local_id], a[base_idx + i]);
    }

    // Reduction to find the maximum
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = WORKGROUP_SIZE_X / 2; stride > 0; stride /= 2)
    {
        if (local_id < stride)
        {
            max_vals[local_id] = max(max_vals[local_id], max_vals[local_id + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    DTYPE max_val = max_vals[0];

    // Compute exponentials and partial sums
    for (uint i = local_id; i < pc.feature_size; i += WORKGROUP_SIZE_X)
    {
        uint idx = base_idx + i;
        DTYPE exp_val = exp(a[idx] - max_val);
        b[idx] = exp_val;
        sum_vals[local_id] += exp_val;
    }

    GroupMemoryBarrierWithGroupSync();
    for (uint stride = WORKGROUP_SIZE_X / 2; stride > 0; stride /= 2)
    {
        if (local_id < stride)
        {
            sum_vals[local_id] += sum_vals[local_id + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    DTYPE sum = sum_vals[0];

    // Normalize by sum
    for (uint i = local_id; i < pc.feature_size; i += WORKGROUP_SIZE_X)
    {
        uint idx = base_idx + i;
        b[idx] /= sum;
    }
}
