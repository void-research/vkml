#ifndef DTYPE
#define DTYPE float
#endif

[vk::constant_id(0)]
const int WORKGROUP_SIZE_X = 1;
[vk::constant_id(1)]
const int WORKGROUP_SIZE_Y = 1;
[vk::constant_id(2)]
const int WORKGROUP_SIZE_Z = 1;

struct PushConstants
{
    uint n;
    uint c;
    uint m;
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
    uint group;
    uint has_bias;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
StructuredBuffer<DTYPE> src;

[[vk::binding(1)]]
StructuredBuffer<DTYPE> weights;

[[vk::binding(2)]]
RWStructuredBuffer<DTYPE> dst;

[[vk::binding(3)]]
StructuredBuffer<DTYPE> bias;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main(uint3 threadId: SV_DispatchThreadID)
{
    uint ox = threadId.x;
    uint oy = threadId.y;
    uint local_z = threadId.z;

    uint out_d_count = pc.out_d;
    uint out_c_count = pc.m;
    uint n_count = pc.n;

    uint od = local_z % out_d_count;
    uint rem = local_z / out_d_count;
    uint oc = rem % out_c_count;
    uint batch = rem / out_c_count;

    if (ox >= pc.out_w || oy >= pc.out_h || od >= pc.out_d || oc >= pc.m || batch >= pc.n)
        return;

    DTYPE acc = DTYPE(0.0);

    uint m_per_group = pc.m / pc.group;
    uint c_per_group = pc.c / pc.group;
    uint group_id = oc / m_per_group;
    uint c_start = group_id * c_per_group;

    for (uint ic = c_start; ic < c_start + c_per_group; ++ic)
    {
        for (uint kd = 0u; kd < pc.k_d; ++kd)
        {
            int in_d_i = int(od) * int(pc.s_d) - int(pc.pad_d) + int(kd) * int(pc.d_d);
            if (in_d_i < 0)
                continue;
            uint in_d_u = uint(in_d_i);
            if (in_d_u >= pc.in_d)
                continue;

            for (uint ky = 0u; ky < pc.k_h; ++ky)
            {
                int in_y_i = int(oy) * int(pc.s_h) - int(pc.pad_h) + int(ky) * int(pc.d_h);
                if (in_y_i < 0)
                    continue;
                uint in_y = uint(in_y_i);
                if (in_y >= pc.in_h)
                    continue;

                for (uint kx = 0u; kx < pc.k_w; ++kx)
                {
                    int in_x_i = int(ox) * int(pc.s_w) - int(pc.pad_w) + int(kx) * int(pc.d_w);
                    if (in_x_i < 0)
                        continue;
                    uint in_x = uint(in_x_i);
                    if (in_x >= pc.in_w)
                        continue;

                    uint src_off = ((((batch * pc.c + ic) * pc.in_d + in_d_u) * pc.in_h + in_y) * pc.in_w) + in_x;
                    uint ic_in_group = ic - c_start;
                    uint w_off = ((((oc * c_per_group + ic_in_group) * pc.k_d + kd) * pc.k_h + ky) * pc.k_w) + kx;

                    acc += src[src_off] * weights[w_off];
                }
            }
        }
    }

    if (pc.has_bias != 0u)
    {
        acc += bias[oc];
    }

    uint dst_off = ((((batch * pc.m + oc) * pc.out_d + od) * pc.out_h + oy) * pc.out_w) + ox;
    dst[dst_off] = acc;
}
