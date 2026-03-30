struct PushConstants {
    uint n;
    uint c;
    uint m;
    uint in_len;
    uint out_len;
    uint kernel;
    uint stride;
    uint dilation;
    uint pad_begin;
    uint group;
    uint has_bias;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
RWStructuredBuffer<DTYPE> src;

[[vk::binding(1)]]
RWStructuredBuffer<DTYPE> weight;

[[vk::binding(2)]]
RWStructuredBuffer<DTYPE> dst;

[[vk::binding(3)]]
RWStructuredBuffer<DTYPE> bias;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main(uint3 threadId: SV_DispatchThreadID) {
    uint ox = threadId.x;
    uint oc = threadId.y;
    uint batch = threadId.z;

    if (ox >= pc.out_len || oc >= pc.m || batch >= pc.n) return;

    DTYPE acc = DTYPE(0.0);

    uint m_per_group = pc.m / pc.group;
    uint c_per_group = pc.c / pc.group;
    uint group_id = oc / m_per_group;
    uint c_start = group_id * c_per_group;

    for (uint ic = c_start; ic < c_start + c_per_group; ++ic) {
        for (uint k = 0u; k < pc.kernel; ++k) {
            int in_x_i = int(ox) * int(pc.stride) - int(pc.pad_begin) + int(k) * int(pc.dilation);
            if (in_x_i < 0) continue;
            uint in_x = uint(in_x_i);
            if (in_x >= pc.in_len) continue;

            uint src_off = ((batch * pc.c + ic) * pc.in_len) + in_x;
            uint ic_in_group = ic - c_start;
            uint w_off = ((oc * c_per_group + ic_in_group) * pc.kernel) + k;

            acc += src[src_off] * weight[w_off];
        }
    }

    if (pc.has_bias != 0u) {
        acc += bias[oc];
    }

    uint dst_off = ((batch * pc.m + oc) * pc.out_len) + ox;
    dst[dst_off] = acc;
}
