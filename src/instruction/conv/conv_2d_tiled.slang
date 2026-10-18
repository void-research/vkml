[vk::constant_id(0)]
const int WORKGROUP_SIZE_X = 1;
[vk::constant_id(1)]
const int WORKGROUP_SIZE_Y = 1;
[vk::constant_id(2)]
const int WORKGROUP_SIZE_Z = 1;

struct PushConstants<T> {
    ImmutablePtr<T> src;
    ImmutablePtr<T> weights;
    Ptr<T> dst;
    ImmutablePtr<T> bias;
    uint n;
    uint c;
    uint m;
    uint in_h;
    uint in_w;
    uint out_h;
    uint out_w;
    uint k_h;
    uint k_w;
    uint s_h;
    uint s_w;
    uint d_h;
    uint d_w;
    uint pad_h;
    uint pad_w;
    uint k;
    uint s;
    uint has_bias;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main<T : IArithmetic>(
    uniform PushConstants<T> pc,
    uint3 groupId: SV_GroupID,
    uint3 threadId: SV_GroupThreadID)
{
    uint batch = groupId.z;
    if (batch >= pc.n)
    {
        return;
    }

    const uint TILE_SIZE = WORKGROUP_SIZE_X;

    uint tile_row = groupId.y;
    uint tile_col = groupId.x;

    uint thread_row = threadId.y;
    uint thread_col = threadId.x;

    uint out_row = tile_row * TILE_SIZE + thread_row;
    uint out_col = tile_col * TILE_SIZE + thread_col;

    T acc = T(0);

    uint num_k_tiles = (pc.k + TILE_SIZE - 1) / TILE_SIZE;

    static groupshared T tileW[WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y];
    static groupshared T tileIn[WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y];

    uint idx = thread_row * TILE_SIZE + thread_col;
    uint k_hw = max(pc.k_h * pc.k_w, 1u);
    uint k_w = max(pc.k_w, 1u);
    uint out_w = max(pc.out_w, 1u);

    for (uint k_tile = 0; k_tile < num_k_tiles; k_tile++)
    {
        uint k_offset = k_tile * TILE_SIZE;

        uint k_col = k_offset + thread_col;
        if (out_row < pc.m && k_col < pc.k)
        {
            uint w_off = out_row * pc.k + k_col;
            tileW[idx] = pc.weights[w_off];
        }
        else
        {
            tileW[idx] = T(0);
        }

        uint k_row = k_offset + thread_row;
        if (k_row < pc.k && out_col < pc.s)
        {
            uint ox = out_col % out_w;
            uint oy = out_col / out_w;

            uint ic = k_row / k_hw;
            uint rem = k_row % k_hw;
            uint ky = rem / k_w;
            uint kx = rem % k_w;

            int in_y_i = int(oy) * int(pc.s_h) - int(pc.pad_h) + int(ky) * int(pc.d_h);
            int in_x_i = int(ox) * int(pc.s_w) - int(pc.pad_w) + int(kx) * int(pc.d_w);

            if (in_y_i >= 0 && in_y_i < int(pc.in_h) && in_x_i >= 0 && in_x_i < int(pc.in_w))
            {
                uint src_off = (((batch * pc.c + ic) * pc.in_h + uint(in_y_i)) * pc.in_w) + uint(in_x_i);
                tileIn[idx] = pc.src[src_off];
            }
            else
            {
                tileIn[idx] = T(0);
            }
        }
        else
        {
            tileIn[idx] = T(0);
        }

        GroupMemoryBarrierWithGroupSync();

        for (uint ki = 0; ki < TILE_SIZE; ki++)
        {
            acc = acc + tileW[thread_row * TILE_SIZE + ki] * tileIn[ki * TILE_SIZE + thread_col];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (out_row < pc.m && out_col < pc.s)
    {
        if (pc.has_bias != 0u)
        {
            acc = acc + pc.bias[out_row];
        }

        uint dst_off = (batch * pc.m + out_row) * pc.s + out_col;
        pc.dst[dst_off] = acc;
    }
}

