struct PushConstants {
    uint m;
    uint k;
    uint n;
    uint stride_a0;
    uint stride_a1;
    uint stride_b0;
    uint stride_b1;
    uint stride_y0;
    uint stride_y1;
    uint trans_a;
    uint trans_b;
    float alpha;
    float beta;
    uint has_c;
};

[[vk::push_constant]]
PushConstants params;

[[vk::binding(0)]]
RWStructuredBuffer<DTYPE> a;

[[vk::binding(1)]]
RWStructuredBuffer<DTYPE> b;

[[vk::binding(2)]]
RWStructuredBuffer<DTYPE> c;

[[vk::binding(3)]]
RWStructuredBuffer<DTYPE> y;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main(uint3 groupId: SV_GroupID, uint3 threadId: SV_GroupThreadID) {
    uint tile_row = groupId.y;
    uint tile_col = groupId.x;

    uint thread_row = threadId.y;
    uint thread_col = threadId.x;

    uint out_row = tile_row * WORKGROUP_SIZE_Y + thread_row;
    uint out_col = tile_col * WORKGROUP_SIZE_X + thread_col;

    DTYPE acc = DTYPE(0.0);

    uint TILE_SIZE = WORKGROUP_SIZE_X; // Assume square tile

    uint stride_a_k = (params.trans_a != 0) ? params.stride_a0 : params.stride_a1;
    uint stride_a_m = (params.trans_a != 0) ? params.stride_a1 : params.stride_a0;
    uint stride_b_k = (params.trans_b != 0) ? params.stride_b1 : params.stride_b0;
    uint stride_b_n = (params.trans_b != 0) ? params.stride_b0 : params.stride_b1;

    uint num_k_tiles = (params.k + TILE_SIZE - 1) / TILE_SIZE;

    if (num_k_tiles == 0) {
        if (out_row < params.m && out_col < params.n) {
            DTYPE result = DTYPE(params.alpha) * DTYPE(0.0);
            if (params.has_c != 0) {
                uint c_idx = out_row * params.n + out_col;
                result += DTYPE(params.beta) * c[c_idx];
            }
            uint y_idx = out_row * params.stride_y0 + out_col * params.stride_y1;
            y[y_idx] = result;
        }
        return;
    }

    static groupshared DTYPE tileA[WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y];
    static groupshared DTYPE tileB[WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y];

    uint idx = thread_row * TILE_SIZE + thread_col;

    for (uint k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        uint k_offset = k_tile * TILE_SIZE;
        uint col_a = k_offset + thread_col;
        uint row_b = k_offset + thread_row;

        tileA[idx] = (out_row < params.m && col_a < params.k)
            ? a[out_row * stride_a_m + col_a * stride_a_k]
            : DTYPE(0.0);
        tileB[idx] = (row_b < params.k && out_col < params.n)
            ? b[row_b * stride_b_k + out_col * stride_b_n]
            : DTYPE(0.0);

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint ki = 0; ki < TILE_SIZE; ki++) {
            acc += tileA[thread_row * TILE_SIZE + ki] * tileB[ki * TILE_SIZE + thread_col];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (out_row < params.m && out_col < params.n) {
        DTYPE result = DTYPE(params.alpha) * acc;
        if (params.has_c != 0) {
            uint c_idx = out_row * params.n + out_col;
            result += DTYPE(params.beta) * c[c_idx];
        }
        uint y_idx = out_row * params.stride_y0 + out_col * params.stride_y1;
        y[y_idx] = result;
    }
}
