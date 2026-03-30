struct PushConstants {
    uint batch_size;
    uint feature_size;
};

[[vk::push_constant]]
PushConstants constants;

[[vk::binding(0)]]
RWStructuredBuffer<DTYPE> a;

[[vk::binding(1)]]
RWStructuredBuffer<DTYPE> b;

groupshared DTYPE subgroup_maxes[64]; // Max supported subgroups
groupshared DTYPE subgroup_sums[64];

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, 1, 1)]
void main(uint3 threadId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    uint local_id = threadId.x;
    uint batch_idx = groupId.x;

    if (batch_idx >= constants.batch_size) {
        return;
    }

    uint base_idx = batch_idx * constants.feature_size;

    // Phase 1: Find maximum using subgroup reduction
    DTYPE local_max = DTYPE(-3.402823466e+38); // -FLT_MAX
    for (uint i = local_id; i < constants.feature_size; i += WORKGROUP_SIZE_X) {
        local_max = max(local_max, a[base_idx + i]);
    }

    // Reduce within subgroup
    DTYPE subgroup_max_val = WaveActiveMax(local_max);

    // One thread per subgroup writes to shared memory
    if (WaveIsFirstLane()) {
        subgroup_maxes[local_id / WaveGetLaneCount()] = subgroup_max_val;
    }

    GroupMemoryBarrierWithGroupSync();

    // Final reduction across subgroups on first invocation
    DTYPE max_val;
    if (local_id == 0) {
        max_val = subgroup_maxes[0];
        uint num_subgroups = (WORKGROUP_SIZE_X + WaveGetLaneCount() - 1) / WaveGetLaneCount();
        for (uint i = 1; i < num_subgroups; i++) {
            max_val = max(max_val, subgroup_maxes[i]);
        }
        subgroup_maxes[0] = max_val;
    }

    GroupMemoryBarrierWithGroupSync();
    max_val = subgroup_maxes[0];

    // Phase 2: Compute exp and sum using subgroup reduction
    DTYPE local_sum = DTYPE(0.0);
    for (uint i = local_id; i < constants.feature_size; i += WORKGROUP_SIZE_X) {
        uint idx = base_idx + i;
        DTYPE exp_val = exp(a[idx] - max_val);
        b[idx] = exp_val; 
        local_sum += exp_val;
    }

    // Reduce sum within subgroup
    DTYPE subgroup_sum_val = WaveActiveSum(local_sum);

    if (WaveIsFirstLane()) {
        subgroup_sums[local_id / WaveGetLaneCount()] = subgroup_sum_val;
    }

    GroupMemoryBarrierWithGroupSync();

    DTYPE sum;
    if (local_id == 0) {
        sum = subgroup_sums[0];
        uint num_subgroups = (WORKGROUP_SIZE_X + WaveGetLaneCount() - 1) / WaveGetLaneCount();
        for (uint i = 1; i < num_subgroups; i++) {
            sum += subgroup_sums[i];
        }
        subgroup_sums[0] = sum;
    }

    GroupMemoryBarrierWithGroupSync();
    sum = subgroup_sums[0];

    // Phase 3: Normalize
    for (uint i = local_id; i < constants.feature_size; i += WORKGROUP_SIZE_X) {
        uint idx = base_idx + i;
        b[idx] /= sum;
    }
}
