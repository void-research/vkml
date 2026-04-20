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
    uint input_len;
    uint output_len;
    uint kernel;
    uint stride;
    uint dilation;
    uint pad_begin;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
StructuredBuffer<DTYPE> src;

[[vk::binding(1)]]
RWStructuredBuffer<DTYPE> dst;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main(uint3 threadId: SV_DispatchThreadID)
{
    uint gid = threadId.x;
    uint total = pc.n * pc.c * pc.output_len;
    if (gid >= total)
        return;

    uint rem = gid;
    uint out_pos = rem % pc.output_len;
    rem /= pc.output_len;
    uint ch = rem % pc.c;
    rem /= pc.c;
    uint batch = rem;

    DTYPE acc = DTYPE(0.0);
    bool found = false;

    for (uint k = 0u; k < pc.kernel; ++k)
    {
        int in_pos_i = int(out_pos) * int(pc.stride) - int(pc.pad_begin) + int(k) * int(pc.dilation);
        if (in_pos_i < 0)
            continue;
        uint in_pos = uint(in_pos_i);
        if (in_pos >= pc.input_len)
            continue;

        uint src_off = ((batch * pc.c + ch) * pc.input_len) + in_pos;
        DTYPE val = src[src_off];
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

    uint dst_off = ((batch * pc.c + ch) * pc.output_len) + out_pos;
    if (found)
        dst[dst_off] = acc;
    else
        dst[dst_off] = DTYPE(0.0);
}
