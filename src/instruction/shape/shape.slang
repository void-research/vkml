[vk::constant_id(0)]
const int WORKGROUP_SIZE_X = 1;
[vk::constant_id(1)]
const int WORKGROUP_SIZE_Y = 1;
[vk::constant_id(2)]
const int WORKGROUP_SIZE_Z = 1;

struct PushConstants
{
    uint slice_len;
    uint start;
    uint pad;
    uint dims_lo[8];
    uint dims_hi[8];
}

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main(
    // We write pairs of uints (low, high) to form 64-bit bounds
    RWStructuredBuffer<uint> dst,
    uint3 threadId: SV_DispatchThreadID)
{
    uint idx = threadId.x;
    if (idx >= pc.slice_len)
    {
        return;
    }

    uint src_idx = pc.start + idx;
    uint low = pc.dims_lo[src_idx];
    uint high = pc.dims_hi[src_idx];

    // Little endian 64-bit word format packing
    uint out_idx = idx * 2u;
    dst[out_idx] = low;
    dst[out_idx + 1u] = high;
}
