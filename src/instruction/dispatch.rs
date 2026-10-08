use crate::gpu::PushConstants;
use crate::slang::Shader;
use crate::tensor_graph::TensorId;
use onnx_extractor::DataType;

pub enum Dispatch {
    Gpu(VkOperation),
    Cpu,
}

pub enum VkOperation {
    Compute {
        shader: &'static Shader,
        dtype: DataType,
        local_size: [u32; 3],
        work_size: [u32; 3],
        push_constants: PushConstants,
        storage_buffers: Vec<Option<TensorId>>,
    },
    Copy {
        src: TensorId,
        dst: TensorId,
    },
}
