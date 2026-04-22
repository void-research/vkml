struct PushConstants
{
    uint total;
}

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
    StructuredBuffer<half> src,
    RWStructuredBuffer<half> dst,
    uint3 threadId: SV_DispatchThreadID)
{
    uint idx = threadId.x;
    if (idx >= pc.total)
        return;

    half val = src[idx];
    dst[idx] = half(1.0) / (half(1.0) + exp(-val));
}
