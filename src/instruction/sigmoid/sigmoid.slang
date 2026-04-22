[vk::constant_id(0)]
const int WORKGROUP_SIZE_X = 1;
[vk::constant_id(1)]
const int WORKGROUP_SIZE_Y = 1;
[vk::constant_id(2)]
const int WORKGROUP_SIZE_Z = 1;

struct PushConstants
{
    uint total;
}

[[vk::push_constant]]
PushConstants pc;

// TODO: FIX, use slang exp() function somehow
T exp_t<T : IFloat>(T x)
{
    T clamped = max(T(-10), min(T(10), x));

    // Approximate e^x as (1 + x / 256)^256 via repeated squaring.
    T y = T(1) + clamped * (T(1) / T(256));
    y = y * y;
    y = y * y;
    y = y * y;
    y = y * y;
    y = y * y;
    y = y * y;
    y = y * y;
    y = y * y;

    return y;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main<T : IFloat>(
    StructuredBuffer<T> src,
    RWStructuredBuffer<T> dst,
    uint3 threadId: SV_DispatchThreadID)
{
    uint idx = threadId.x;
    if (idx >= pc.total)
        return;

    T val = src[idx];
    dst[idx] = T(1) / (T(1) + exp_t(T(0) - val));
}
