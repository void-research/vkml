use crate::gpu::PushConstants;
use crate::tensor_graph::TensorId;

pub enum Dispatch {
    Gpu(VkOperation),
    Cpu,
}

pub enum VkOperation {
    Compute {
        spv: &'static [u8],
        local_size: [u32; 3],
        work_size: [u32; 3],
        push_constants: PushConstants,
        storage_buffers: Box<[Option<TensorId>]>,
    },
    ClearAndCompute {
        clear_dst: TensorId,
        spv: &'static [u8],
        local_size: [u32; 3],
        work_size: [u32; 3],
        push_constants: PushConstants,
        storage_buffers: Box<[Option<TensorId>]>,
    },
    Copy {
        src: TensorId,
        dst: TensorId,
    },
}
