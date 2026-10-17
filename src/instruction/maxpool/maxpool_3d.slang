[vk::constant_id(0)]
const int WORKGROUP_SIZE_X = 1;
[vk::constant_id(1)]
const int WORKGROUP_SIZE_Y = 1;
[vk::constant_id(2)]
const int WORKGROUP_SIZE_Z = 1;

struct PushConstants<T> {
    ImmutablePtr<T> src;
    Ptr<T> dst;
    uint n;
    uint c;
    uint in_d;
    uint in_h;
    uint in_w;
    uint out_d;
    uint out_h;
    uint out_w;
    uint k_d;
    uint k_h;
    uint k_w;
    uint s_d;
    uint s_h;
    uint s_w;
    uint d_d;
    uint d_h;
    uint d_w;
    uint pad_d;
    uint pad_h;
    uint pad_w;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main<T : IArithmetic>(
    uniform PushConstants<T> pc,
    uint3 threadId: SV_DispatchThreadID)
{
    uint out_x = threadId.x;
    uint out_y = threadId.y;
    uint z_index = threadId.z; // encodes (d * n * c) + (batch * c + ch)

    if (out_x >= pc.out_w || out_y >= pc.out_h)
        return;

    uint total_z = pc.out_d * pc.n * pc.c;
    if (z_index >= total_z)
        return;

    uint out_d_idx = z_index / (pc.n * pc.c);
    uint rem = z_index % (pc.n * pc.c);
    uint batch = rem / pc.c;
    uint ch = rem % pc.c;

    T acc = T(0);
    bool found = false;

    for (uint kd = 0u; kd < pc.k_d; ++kd)
    {
        int in_d_i = int(out_d_idx) * int(pc.s_d) - int(pc.pad_d) + int(kd) * int(pc.d_d);
        if (in_d_i < 0)
            continue;
        uint in_d = uint(in_d_i);
        if (in_d >= pc.in_d)
            continue;

        for (uint ky = 0u; ky < pc.k_h; ++ky)
        {
            int in_y_i = int(out_y) * int(pc.s_h) - int(pc.pad_h) + int(ky) * int(pc.d_h);
            if (in_y_i < 0)
                continue;
            uint in_y = uint(in_y_i);
            if (in_y >= pc.in_h)
                continue;

            for (uint kx = 0u; kx < pc.k_w; ++kx)
            {
                int in_x_i = int(out_x) * int(pc.s_w) - int(pc.pad_w) + int(kx) * int(pc.d_w);
                if (in_x_i < 0)
                    continue;
                uint in_x = uint(in_x_i);
                if (in_x >= pc.in_w)
                    continue;

                uint src_off = ((((batch * pc.c + ch) * pc.in_d + in_d) * pc.in_h + in_y) * pc.in_w) + in_x;
                T val = pc.src[src_off];
                if (!found)
                {
                    acc = val;
                    found = true;
                }
                else if (val > acc)
                {
                    acc = val;
                }
            }
        }
    }

    uint dst_off = ((((batch * pc.c + ch) * pc.out_d + out_d_idx) * pc.out_h + out_y) * pc.out_w) + out_x;
    if (found)
        pc.dst[dst_off] = acc;
    else
        pc.dst[dst_off] = T(0);
}
