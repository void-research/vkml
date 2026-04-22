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
}

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(32, 32, 1)]
void main<T : IArithmetic>(
    StructuredBuffer<T> src1,
    StructuredBuffer<T> src2,
    RWStructuredBuffer<T> dst,
    uint3 groupId: SV_GroupID, uint3 threadId: SV_GroupThreadID)
{
    uint tile_row = groupId.y;
    uint tile_col = groupId.x;

    uint thread_row = threadId.y;
    uint thread_col = threadId.x;

    const uint TILE_SIZE = 32;
    uint out_row = tile_row * TILE_SIZE + thread_row;
    uint out_col = tile_col * TILE_SIZE + thread_col;

    T acc = T(0);

    uint num_k_tiles = (pc.k + TILE_SIZE - 1) / TILE_SIZE;

    if (num_k_tiles == 0)
    {
        if (out_row < pc.m && out_col < pc.n)
        {
            dst[out_row * pc.stride_c0 + out_col * pc.stride_c1] = T(0);
        }
        return;
    }

    static groupshared T tileA[TILE_SIZE * TILE_SIZE];
    static groupshared T tileB[TILE_SIZE * TILE_SIZE];

    uint idx = thread_row * TILE_SIZE + thread_col;

    for (uint k_tile = 0; k_tile < num_k_tiles; k_tile++)
    {
        uint k_offset = k_tile * TILE_SIZE;
        uint col_a = k_offset + thread_col;
        uint row_b = k_offset + thread_row;

        tileA[idx] = (out_row < pc.m && col_a < pc.k)
                         ? src1[out_row * pc.stride_a0 + col_a * pc.stride_a1]
                         : T(0);

        tileB[idx] = (row_b < pc.k && out_col < pc.n)
                         ? src2[row_b * pc.stride_b0 + out_col * pc.stride_b1]
                         : T(0);

        GroupMemoryBarrierWithGroupSync();

        for (uint ki = 0; ki < TILE_SIZE; ki++)
        {
            acc = acc + tileA[thread_row * TILE_SIZE + ki] * tileB[ki * TILE_SIZE + thread_col];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (out_row < pc.m && out_col < pc.n)
    {
        dst[out_row * pc.stride_c0 + out_col * pc.stride_c1] = acc;
    }
}
