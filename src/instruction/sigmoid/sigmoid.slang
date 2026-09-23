struct PushConstants
{
    uint total;
}

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main<T : __BuiltinFloatingPointType>(
    StructuredBuffer<T> src,
    RWStructuredBuffer<T> dst,
    uint3 threadId: SV_DispatchThreadID)
{
    uint idx = threadId.x;
    if (idx >= pc.total)
        return;

    T val = src[idx];
    dst[idx] = T(1.0) / (T(1.0) + exp(-val));
}

