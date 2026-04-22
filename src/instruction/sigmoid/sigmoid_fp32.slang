struct PushConstants
{
    uint total;
}

[[vk::push_constant]]
PushConstants pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
    StructuredBuffer<float> src,
    RWStructuredBuffer<float> dst,
    uint3 threadId: SV_DispatchThreadID)
{
    uint idx = threadId.x;
    if (idx >= pc.total)
        return;

    float val = src[idx];
    dst[idx] = 1.0f / (1.0f + exp(-val));
}
