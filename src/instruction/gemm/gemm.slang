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
    uint stride_y0;
    uint stride_y1;
    uint trans_a;
    uint trans_b;
    float alpha;
    float beta;
    uint has_c;
}

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main<T : IArithmetic>(
    StructuredBuffer<T> a,
    StructuredBuffer<T> b,
    StructuredBuffer<T> c,
    RWStructuredBuffer<T> y,
    uint3 threadId: SV_DispatchThreadID)
{
    uint row = threadId.y;
    uint col = threadId.x;

    if (row >= pc.m || col >= pc.n)
    {
        return;
    }

    T sum = T(0);

    uint stride_a_k = (pc.trans_a != 0) ? pc.stride_a0 : pc.stride_a1;
    uint stride_a_m = (pc.trans_a != 0) ? pc.stride_a1 : pc.stride_a0;
    uint stride_b_k = (pc.trans_b != 0) ? pc.stride_b1 : pc.stride_b0;
    uint stride_b_n = (pc.trans_b != 0) ? pc.stride_b0 : pc.stride_b1;

    for (uint i = 0; i < pc.k; i++)
    {
        uint a_idx = row * stride_a_m + i * stride_a_k;
        uint b_idx = i * stride_b_k + col * stride_b_n;
        sum = sum + a[a_idx] * b[b_idx];
    }

    T result = T(pc.alpha) * sum;

    if (pc.has_c != 0)
    {
        uint c_idx = row * pc.n + col;
        result = result + T(pc.beta) * c[c_idx];
    }

    uint y_idx = row * pc.stride_y0 + col * pc.stride_y1;
    y[y_idx] = result;
}
