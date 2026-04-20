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
    uint n;
    uint stride_a0;
    uint stride_a1;
    uint stride_b0;
    uint stride_b1;
    uint stride_c0;
    uint stride_c1;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
RWStructuredBuffer<DTYPE> a;

[[vk::binding(1)]]
RWStructuredBuffer<DTYPE> b;

[[vk::binding(2)]]
RWStructuredBuffer<DTYPE> c;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main(uint3 groupId: SV_GroupID, uint3 threadId: SV_GroupThreadID)
{
    uint tile_row = groupId.y;
    uint tile_col = groupId.x;

    uint thread_row = threadId.y;
    uint thread_col = threadId.x;

    uint out_row = tile_row * WORKGROUP_SIZE_Y + thread_row;
    uint out_col = tile_col * WORKGROUP_SIZE_X + thread_col;

    DTYPE acc = DTYPE(0.0);

    uint TILE_SIZE = WORKGROUP_SIZE_X; // Assume square tile
    uint num_k_tiles = (pc.k + TILE_SIZE - 1) / TILE_SIZE;

    if (num_k_tiles == 0)
    {
        if (out_row < pc.m && out_col < pc.n)
        {
            c[out_row * pc.stride_c0 + out_col * pc.stride_c1] = DTYPE(0.0);
        }
        return;
    }

    static groupshared DTYPE tileA[WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y];
    static groupshared DTYPE tileB[WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y];

    uint idx = thread_row * WORKGROUP_SIZE_X + thread_col;

    for (uint k_tile = 0; k_tile < num_k_tiles; k_tile++)
    {
        uint k_offset = k_tile * TILE_SIZE;
        uint col_a = k_offset + thread_col;
        uint row_b = k_offset + thread_row;

        tileA[idx] = (out_row < pc.m && col_a < pc.k)
                         ? a[out_row * pc.stride_a0 + col_a * pc.stride_a1]
                         : DTYPE(0.0);

        tileB[idx] = (row_b < pc.k && out_col < pc.n)
                         ? b[row_b * pc.stride_b0 + out_col * pc.stride_b1]
                         : DTYPE(0.0);

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint ki = 0; ki < TILE_SIZE; ki++)
        {
            acc += tileA[thread_row * TILE_SIZE + ki] * tileB[ki * TILE_SIZE + thread_col];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (out_row < pc.m && out_col < pc.n)
    {
        c[out_row * pc.stride_c0 + out_col * pc.stride_c1] = acc;
    }
}
