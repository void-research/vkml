struct PC
{
    uint total;
}

[[vk::push_constant]]
ConstantBuffer<PC> pc;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
void main(
    StructuredBuffer<DTYPE> a,
    RWStructuredBuffer<DTYPE> b,
    uint3 threadId: SV_DispatchThreadID)
{
    uint idx = threadId.x;
    if (idx >= pc.total)
        return;

    DTYPE val = a[idx];
    b[idx] = (DTYPE)1 / ((DTYPE)1 + exp(-val));
}
