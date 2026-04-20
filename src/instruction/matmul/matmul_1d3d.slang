[vk::constant_id(0)]
const int WORKGROUP_SIZE_X = 1;
[vk::constant_id(1)]
const int WORKGROUP_SIZE_Y = 1;
[vk::constant_id(2)]
const int WORKGROUP_SIZE_Z = 1;

struct PushConstants
{
    uint batch;
    uint k;
    uint n;
    uint stride_a;
    uint stride_b0;
    uint stride_b1;
    uint stride_b2;
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
void main(uint3 threadId: SV_DispatchThreadID)
{
    uint col = threadId.x;
    uint batch = threadId.y;

    if (col >= pc.n || batch >= pc.batch)
    {
        return;
    }

    DTYPE sum = DTYPE(0.0);

    for (uint i = 0; i < pc.k; i++)
    {
        uint a_idx = i * pc.stride_a;
        uint b_idx = batch * pc.stride_b0 + i * pc.stride_b1 + col * pc.stride_b2;
        sum += a[a_idx] * b[b_idx];
    }

    uint c_idx = batch * pc.stride_c0 + col * pc.stride_c1;
    c[c_idx] = sum;
}
